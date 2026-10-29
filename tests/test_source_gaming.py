"""FIX_CODE repairs are verified by the untouched tests, so a source patch must not be able to detect the test run.

Real ``validate_patch`` on a real git fixture (``tools.patch_content`` source checks).  Each attack would make the
original tests pass only under pytest; each legitimate change must stay editable.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from testorcode.schemas.episode_schemas import EditPolicy, PatchScope
from testorcode.tools.patch_tools import validate_patch

POLICY = EditPolicy(allowed_source_roots=("pkg/",), allowed_test_roots=("tests/",))
CORE = "import os\n\n\ndef norm(s):\n    \"\"\"Lower-case s (tested with pytest).\"\"\"\n    return s.capitalize()\n"


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "r"
    for rel, text in {"pkg/__init__.py": "", "pkg/core.py": CORE,
                      "pkg/frames.py": "import sys\n\n\ndef caller():\n    return sys._getframe(1).f_code.co_name\n",
                      "tests/test_core.py": "from pkg.core import norm\n\n\ndef test_norm():\n    assert norm('a') == 'a'\n"
                      }.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", "c")
    return root


def source_patch(root: Path, new_core: str, path: str = "pkg/core.py") -> str:
    target = root / path
    old = target.read_text()
    target.write_text(new_core)
    diff = git(root, "diff")
    target.write_text(old)
    return diff


GAMED_BODY = "    return s.lower() if {cond} else s.capitalize()\n"
ATTACKS = {
    "sys.modules": ("import sys\n", "'pytest' in sys.modules"),
    "aliased sys.modules": ("import sys as _s\n", "len(_s.modules) > 400"),
    "from sys import modules": ("from sys import modules\n", "len(modules) > 400"),
    "getattr(sys, 'modules')": ("import sys\n", "len(getattr(sys, 'modules')) > 400"),
    "PYTEST_CURRENT_TEST env": ("", "os.environ.get('PYTEST_CURRENT_TEST')"),
    "pytest in argv": ("import sys\n", "'pytest' in sys.argv[0]"),
    "sys._getframe": ("import sys\n", "sys._getframe(1).f_code.co_name.startswith('test_')"),
    "inspect.stack": ("import inspect\n", "any(f.function.startswith('test_') for f in inspect.stack())"),
    "aliased inspect": ("import inspect as ins\n", "bool(ins.stack())"),
    "from inspect import currentframe": ("from inspect import currentframe\n", "bool(currentframe())"),
    "traceback.extract_stack": ("import traceback\n", "'test_' in str(traceback.extract_stack())"),
    "sys.gettrace (coverage/debugger)": ("import sys\n", "sys.gettrace() is not None"),
    "frame f_back": ("import inspect\n", "'test' in str(inspect.currentframe().f_back.f_locals)"),
    "import pytest": ("import pytest\n", "True"),
    "__import__ of pytest internals": ("", "__import__('_pytest') is not None"),
    "pytest hook in source": ("", "True", "\n\ndef pytest_configure(config):\n    pass\n"),
}


@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_source_patches_that_detect_the_test_run_are_rejected(repo: Path, name: str) -> None:
    imports, cond, *suffix = ATTACKS[name]
    new = CORE.replace("import os\n", f"import os\n{imports}").replace(
        "    return s.capitalize()\n", GAMED_BODY.format(cond=cond)) + "".join(suffix)
    result = validate_patch(repo, source_patch(repo, new), policy=POLICY, scope=PatchScope.SRC)
    assert not result.ok, name
    assert any("introduces" in r for r in result.rejection_reasons), result.rejection_reasons


@pytest.mark.parametrize(
    ("path", "new"),
    [
        ("pkg/core.py", CORE.replace("s.capitalize()", "s.lower()")),  # the real fix
        ("pkg/core.py", CORE.replace("s.capitalize()", "s.strip().lower() if s else s")),
        ("pkg/core.py", CORE.replace("import os\n", "import os\nimport sys\n").replace(
            "s.capitalize()", "s.lower() if sys.version_info >= (3, 8) else s.lower()")),  # sys use, not detection
        ("pkg/core.py", CORE.replace("(tested with pytest)", "(tested with pytest, see tests/)")),  # docstring only
        ("pkg/core.py", CORE.replace("s.capitalize()", "os.fspath(s).lower()")),
        # A file that already uses frames stays editable: only constructs a patch *adds* are judged.
        ("pkg/frames.py", "import sys\n\n\ndef caller():\n    return sys._getframe(2).f_code.co_name\n"),
    ],
)
def test_legitimate_source_fixes_stay_editable(repo: Path, path: str, new: str) -> None:
    result = validate_patch(repo, source_patch(repo, new, path), policy=POLICY, scope=PatchScope.SRC)
    assert result.ok, result.rejection_reasons
