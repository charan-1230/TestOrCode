"""Process isolation for running repository tests.

Three modes:

* ``docker``  - ``docker run`` with no network, a read-only root filesystem,
  the worktree mounted read-only, all capabilities dropped, no new
  privileges, a non-root user and pid/memory/cpu limits.  Needs an image that
  already contains Python, pytest and the repository's dependencies
  (``EnvironmentSpec.docker_image``).
* ``bwrap``   - bubblewrap (unprivileged Linux namespaces): no network, its own
  pid namespace (everything dies with it), only ``/usr``, the Python
  environment and the worktree (read-only) are visible, ``/tmp`` is private,
  environment cleared.  No cgroup limits; rlimits only.
* ``trusted-host`` - a plain subprocess on the host.  For trusted local
  fixtures only; it must be requested explicitly.

``auto`` (the default) picks docker when an image is given and the daemon is
reachable, else bwrap when it works, else raises ``ToolError``: untrusted code
is never run on the host by default.

What this does **not** guarantee: tests run inside the sandbox and can still
falsify their own results (they share the pytest process and can write the
report).  The runner cross-checks the report but cannot prove it honest.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from testorcode.errors import ToolError
from testorcode.logging_config import get_logger

logger = get_logger(__name__)

DOCKER_WORKDIR: Final[str] = "/work"
DOCKER_OUTDIR: Final[str] = "/out"
DOCKER_PROBEDIR: Final[str] = "/probe"
DOCKER_LIMITS: Final[tuple[str, ...]] = (
    "--pids-limit",
    "256",
    "--memory",
    "4g",
    "--memory-swap",
    "4g",
    "--cpus",
    "2",
    "--tmpfs",
    "/tmp:rw,nosuid,nodev,size=512m",
)
RLIMITS_BYTES: Final[dict[str, int]] = {
    "RLIMIT_FSIZE": 1 << 30,  # 1 GiB per file
    "RLIMIT_AS": 8 << 30,  # 8 GiB address space
}
_HOST_LIB_DIRS: Final[tuple[str, ...]] = ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32")
_ETC_FILES: Final[tuple[str, ...]] = ("/etc/ld.so.cache", "/etc/localtime", "/etc/alternatives")


class Isolation(StrEnum):
    AUTO = "auto"
    DOCKER = "docker"
    BWRAP = "bwrap"
    TRUSTED_HOST = "trusted-host"


@dataclass(frozen=True)
class Layout:
    """Where the worktree, output and probe directories are, as seen by pytest."""

    worktree: str
    outdir: str
    probedir: str
    python: str


@functools.cache
def docker_available() -> bool:
    """The docker CLI exists and the daemon answers (without sudo)."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


@functools.cache
def bwrap_available() -> bool:
    """bubblewrap exists and can create unprivileged namespaces here."""
    if shutil.which("bwrap") is None or not Path("/usr/bin/true").exists():
        return False
    try:
        result = subprocess.run(
            [*_bwrap_base(), "--", "/usr/bin/true"],
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def resolve_isolation(isolation: Isolation | str, docker_image: str | None) -> Isolation:
    """Turn ``auto`` into a concrete mode, or fail closed.

    Raises:
        ValueError: unknown mode.
        ToolError: the requested sandbox is not available.
    """
    mode = Isolation(isolation)
    if mode is Isolation.TRUSTED_HOST:
        logger.warning("running tests on the host without isolation (trusted fixtures only)")
        return mode
    if mode is Isolation.DOCKER:
        if not docker_image:
            raise ToolError("docker isolation needs a docker image (EnvironmentSpec.docker_image)")
        if not docker_available():
            raise ToolError("docker isolation requested but the docker daemon is not reachable")
        return mode
    if mode is Isolation.BWRAP:
        if not bwrap_available():
            raise ToolError("bwrap isolation requested but bubblewrap is unavailable")
        return mode
    if docker_image and docker_available():
        return Isolation.DOCKER
    if bwrap_available():
        return Isolation.BWRAP
    raise ToolError(
        "no sandbox available (docker unreachable or no image, bubblewrap unusable); "
        "refusing to run repository tests on the host. Use isolation='trusted-host' "
        "only for trusted local fixtures."
    )


def layout_for(
    mode: Isolation, worktree: Path, outdir: Path, probedir: Path, python: str
) -> Layout:
    if mode is Isolation.DOCKER:
        return Layout(DOCKER_WORKDIR, DOCKER_OUTDIR, DOCKER_PROBEDIR, python)
    return Layout(str(worktree), str(outdir), str(probedir), python)


def wrap_command(
    mode: Isolation,
    inner: list[str],
    *,
    worktree: Path,
    outdir: Path,
    probedir: Path,
    python: str,
    env: Mapping[str, str],
    docker_image: str | None = None,
    container_name: str | None = None,
) -> tuple[list[str], dict[str, str] | None]:
    """Command to execute plus the environment for ``Popen`` (None = inherit nothing extra).

    For the sandboxes the child's environment is passed *inside* the sandbox
    (``--setenv`` / ``-e``); the sandbox launcher itself gets a minimal one.
    """
    if mode is Isolation.TRUSTED_HOST:
        return inner, dict(env)
    launcher_env = {"PATH": os.environ.get("PATH", os.defpath), "LC_ALL": "C"}
    if mode is Isolation.BWRAP:
        args = _bwrap_base()
        for path in _python_mounts(python):
            args += ["--ro-bind", path, path]
        args += [
            "--ro-bind",
            str(probedir),
            str(probedir),
            "--ro-bind",
            str(worktree),
            str(worktree),
            "--bind",
            str(outdir),
            str(outdir),
            "--chdir",
            str(worktree),
            "--clearenv",
        ]
        for key, value in env.items():
            args += ["--setenv", key, value]
        return [*rlimit_prefix(), *args, "--", *inner], launcher_env
    if mode is Isolation.DOCKER:
        if not docker_image or not container_name:
            raise ToolError("docker isolation needs an image and a container name")
        uid, gid = os.getuid(), os.getgid()
        user = "65534:65534" if uid == 0 else f"{uid}:{gid}"
        args = [
            "docker",
            "run",
            "--rm",
            "--init",
            "--name",
            container_name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            user,
            *DOCKER_LIMITS,
            "--mount",
            f"type=bind,src={worktree},dst={DOCKER_WORKDIR},readonly",
            "--mount",
            f"type=bind,src={probedir},dst={DOCKER_PROBEDIR},readonly",
            "--mount",
            f"type=bind,src={outdir},dst={DOCKER_OUTDIR}",
            "--workdir",
            DOCKER_WORKDIR,
        ]
        for key, value in env.items():
            args += ["--env", f"{key}={value}"]
        return [*args, docker_image, *inner], launcher_env
    raise ValueError(f"unresolved isolation mode {mode}")


def stop_container(container_name: str) -> None:
    """Kill and remove a container (best effort; used on timeout and cleanup)."""
    for args in (["kill", container_name], ["rm", "-f", container_name]):
        try:
            subprocess.run(["docker", *args], capture_output=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired):
            logger.warning("docker %s failed for a test container", args[0])


def rlimit_prefix() -> list[str]:
    """``prlimit`` prefix for the bwrap launcher: no core dumps, bounded files and memory.

    Empty (and logged) when util-linux ``prlimit`` is missing; a ``preexec_fn``
    is avoided because it is unsafe in threaded callers.
    """
    if shutil.which("prlimit") is None:
        logger.warning("prlimit not found: bwrap sandbox runs without rlimits")
        return []
    return [
        "prlimit",
        "--core=0",
        f"--fsize={RLIMITS_BYTES['RLIMIT_FSIZE']}",
        f"--as={RLIMITS_BYTES['RLIMIT_AS']}",
        "--",
    ]


def _bwrap_base() -> list[str]:
    args = [
        "bwrap",
        "--unshare-all",  # network, pid, ipc, uts, user, cgroup namespaces
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--hostname",
        "testorcode-sandbox",
        "--ro-bind",
        "/usr",
        "/usr",
    ]
    for directory in _HOST_LIB_DIRS:
        if os.path.islink(directory):
            args += ["--symlink", os.readlink(directory), directory]
        elif os.path.isdir(directory):
            args += ["--ro-bind", directory, directory]
    for path in _ETC_FILES:
        args += ["--ro-bind-try", path, path]
    # /tmp first, so that binds of paths under /tmp (worktrees, envs) land on top of it.
    args += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/tmp/home"]
    return args


def _python_mounts(python: str) -> list[str]:
    """Read-only mounts that make ``python`` runnable (its venv or the file itself)."""
    exe = Path(python)
    if not exe.is_absolute():
        found = shutil.which(python)
        if found is None:
            return []
        exe = Path(found)
    mounts: list[str] = []
    env_root = exe.parent.parent
    if (env_root / "pyvenv.cfg").is_file():
        mounts.append(str(env_root))
    else:
        mounts.append(str(exe))
    real = exe.resolve()
    if real != exe:
        mounts.append(str(real))
    return [m for m in dict.fromkeys(mounts) if not (m == "/usr" or m.startswith("/usr/"))]
