# TestOrCode

**Intent-Grounded Triage of CI Test Failures with Scoped Change Memory and Test-Integrity Checking**

TestOrCode is a research-oriented AI agent system designed to analyse CI test failures, identify likely causes, recommend or implement scoped repairs, and verify that proposed changes preserve test integrity.

The system uses four logical agent roles coordinated by a Python orchestrator. It combines intent-grounded retrieval, scoped change memory, executable verification, and an independent test-integrity guard.

## 1. Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Project and dependency management | `pip` + `pyproject.toml` |
| Dependency locking | `pip-tools` + `requirements-lock.txt` |
| Testing | `pytest` |
| Code linting | Ruff |
| Structured outputs | Pydantic |
| LLM integration | OpenAI SDK / compatible hosted API |
| Initial retrieval | `rank-bm25` |
| Semantic embeddings | `sentence-transformers` (optional dependency group) |
| Memory and episode storage | JSON / JSONL |
| Experiment analysis | pandas |
| Git history and repository operations | Git + Python `subprocess` |
| Mutation testing | `mutmut` |
| Execution sandbox | Docker + Git worktrees |

**Dependency management policy:** We use `pip` and `pyproject.toml`. Do not introduce `uv` into the project setup. The committed `requirements-lock.txt` is the shared source of resolved Python package versions.

## 2. Prerequisites

Every team member should have:

- Python 3.12 available on their system.
- Git installed and configured.
- Docker installed and running for sandbox-related tasks.
- Access to the project repository.
- Access to the selected hosted LLM API when working on live model integration.

Use Python 3.12 for this project. Ubuntu 22.04 users may have Python 3.10 as their system Python; always use the project's Python 3.12 virtual environment rather than modifying the system Python.

## 3. Clone the Repository

Clone the main repository and enter the project directory:

```bash
git clone <REPOSITORY_URL>
cd testorcode
```

Replace `<REPOSITORY_URL>` with the actual repository URL.

Ensure that the repository contains `pyproject.toml` and the committed `requirements-lock.txt`.

## 4. Create and Activate the Virtual Environment

Each developer maintains their own local virtual environment. The environment directory must not be committed to Git.

Create it using Python 3.12:

```bash
python3.12 -m venv myvenv
```

Activate it:

```bash
source myvenv/bin/activate
```

Verify the interpreter and pip:

```bash
python --version
python -m pip --version
```

The Python version should be 3.12.x, and the pip path should point inside `.venv`.

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

## 5. Install Project Dependencies

The project dependencies are declared in `pyproject.toml`. Exact resolved package versions are recorded in `requirements-lock.txt`.

Install the locked dependencies:

```bash
python -m pip install -r requirements-lock.txt
```

Install the project itself in editable mode:

```bash
python -m pip install --no-deps -e .
```

The editable installation allows changes to the source code under `src/testorcode` to be used without reinstalling the project after every code change.

**Important:** Team members should not independently regenerate the lock file during normal setup. Use the committed lock file so that everyone installs the same resolved dependency versions.

## 6. Verify the Installation

Run the following commands from the project root with `myvenv` activated:

```bash
python --version
python -m pip check
python -m pytest --version
python -m ruff --version
python -c "import pydantic, pandas; print('Core imports OK')"
```

Run the project test suite:

```bash
python -m pytest
```

If a test fails, review the failure output before proceeding. Do not assume that an installation is successful merely because pip completed.

## 7. Install and Verify System Tools

Git and Docker are system-level tools, not Python dependencies. They are not installed by `pyproject.toml` or the Python lock file.

Install Git and docker:
```bash
sudo apt update
sudo apt install git docker.io
```

Verify Git:

```bash
git --version
```

Verify Docker:

```bash
docker --version
docker info
```

`docker info` should succeed when Docker is running and your user has permission to access the Docker daemon.

Git worktrees can be used to create isolated working directories for different branches (applicable only if one person is working on different branches simultaneously):

```bash
git worktree list
```

Do not grant an AI repair agent unrestricted access to the host shell. Repository paths, allowed commands, execution limits, and proposed patches must be validated by the host-side orchestrator.

## 8. Environment Variables and API Keys

Copy `.env.example` to `.env` :

```bash
cp .env.example .env
```

Add the required API configuration to `.env` according to the selected provider.

The configuration used for the Day-1 runs (Gemini through its OpenAI-compatible endpoint; no extra SDK):

```dotenv
LLM_PROVIDER=gemini
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-3.5-flash
LLM_STRUCTURED_MODE=json_schema
LLM_MAX_RETRIES=6
LLM_TIMEOUT_SECONDS=90
# Interpreter for the target repository's tests (see section 8a); the venv path, not its resolved symlink.
TEST_PYTHON=/abs/path/to/boltons-venv/bin/python
```

`gemini-2.5-flash` is no longer available to new keys. The free tier allows about 20 requests per day per model
and per Google Cloud project, and often answers 503 under load; the client retries those. One episode takes 2-4 calls.

Never commit `.env`, API keys, access tokens, or other secrets to Git. Keep `.env.example` free of real credentials.

The hosted model should be accessed through one common client interface. Before selecting the final model, verify structured output support, tool-calling behavior, availability, rate limits, and API access.

## 8a. Run a Pilot Episode End to End

The pilot episodes run against a pinned clone of boltons with its own interpreter (the dataset was verified with
pytest 8.4.2; this project's lock pins a newer pytest). From the project root:

```bash
git clone https://github.com/mahmoud/boltons ../boltons
git -C ../boltons checkout d31d821d56c8f7756af9c30a864077f2e5855321
python3.12 -m venv ../boltons-venv && ../boltons-venv/bin/python -m pip install pytest==8.4.2

python scripts/llm_smoke.py --episode dev-05          # one live model call, validated as IntentAnalysis
python scripts/run_episode.py --episode dev-05        # failure -> decision -> patch/escalation -> rerun
```

`run_episode.py` materialises the episode's CI state in `../episode-worktrees/<episode>`, runs the tests in the
bubblewrap sandbox, and writes `results/day1/<run_id>.json` (the `ExperimentRunRecord`) plus a `.summary.json` that
scores the prediction against the gold label after the run. Runs are recorded as condition A2 (no memory) until the
memory store exists. If your shell exports a `PYTHONPATH` (e.g. ROS), `unset PYTHONPATH` first; the sandbox ignores it
but direct `pytest` calls do not. Results and verdicts so far: [`docs/day1_overview.md`](docs/day1_overview.md).

## 9. Project Structure

The repository is organised around a Python package, tests, evaluation datasets, and experiment results.

```text
testorcode/
├── pyproject.toml
├── requirements-lock.txt
├── README.md
├── .env.example
├── .gitignore
├── docs/
│   └── interfaces.md          # shared interfaces: read before branching
├── src/
│   └── testorcode/
│       ├── __init__.py
│       ├── config.py          # Settings.from_env()
│       ├── logging_config.py
│       ├── errors.py
│       ├── schemas/           # episode_schemas, result_schemas, contracts
│       ├── orchestrator.py
│       ├── llm_client.py
│       ├── agents/
│       │   ├── triage.py
│       │   ├── intent_analyst.py
│       │   ├── repairer.py
│       │   └── integrity_guard.py
│       ├── tools/
│       │   ├── pytest_runner.py
│       │   ├── git_tools.py
│       │   └── patch_tools.py
│       ├── retrieval.py
│       ├── memory.py
│       └── evaluation.py
├── datasets/
├── tests/
├── results/
└── scripts/
```

Most modules are interface stubs that raise `NotImplementedError`; `docs/interfaces.md` says which parts are stable.

The exact module layout may evolve as implementation progresses. The team should keep shared schemas and interfaces consistent.

## 10. Development Workflow

### Branching

Use feature branches rather than making all changes directly on `main`.

Suggested branches:

- `agent-core` — orchestration, LLM client, schemas, and agent responsibilities.
- `sandbox` — Git operations, test execution, patch handling, and sandbox isolation.
- `evaluation` — episode datasets, experiment runner, metrics, and result analysis.


### Before committing

Activate `myvenv`, then run:

```bash
python -m pytest
python -m ruff check .
git status
```

Review the changes and commit only the intended files.

```bash
git add <files>
git commit -m "Describe the change"
git push
```

Use pull requests or coordinated merges to integrate changes into `main`. Never commit virtual environments, API keys, temporary model files, or unrelated local changes.

## 11. Dependency Management Policy

### Adding or changing dependencies

1. Discuss the dependency with the team.
2. Update `pyproject.toml` with the required dependency and appropriate version constraint.
3. Regenerate `requirements-lock.txt` in the project's Python 3.12 environment.
4. Install the updated dependencies and run tests.
5. Commit both files together.

To install the dependency-locking tool when needed:

```bash
python -m pip install "pip-tools>=7.4,<8"
```

To regenerate the lock file after an intentional dependency change:

```bash
python -m piptools compile \
  --all-extras \
  pyproject.toml \
  --output-file requirements-lock.txt
```

Review the changes to the generated file before committing them.

To update the current environment from the new lock file:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pip check
```

Do not regenerate the lock file simply to refresh packages without team agreement. A dependency update can change behavior and invalidate previous experimental results.

### Reproducibility

The team should commit both `pyproject.toml` and `requirements-lock.txt`.

The former defines project requirements; the latter records the resolved package versions. 

## 12. Research and Evaluation Principles

The implementation should preserve the research goals of TestOrCode:

- Implement the four logical agent roles and their orchestration.
- Support `UPDATE_TEST`, `FIX_CODE`, and `ESCALATE` decisions.
- Retrieve relevant evidence using BM25 before introducing more complex retrieval.
- Store scoped memory in JSON/JSONL and validate reusable records with executable probes.
- Verify proposed changes using test execution and an independent test-integrity guard.
- Implement the Full, B1, A1, A2, and A3 experimental conditions using configuration switches.
- Keep gold labels and reference repairs in the evaluation harness, separate from agent-visible inputs.
- Record experiment conditions, seeds, model IDs, predictions, retrieved evidence, patches, test outcomes, integrity results, token usage, tool calls, and errors.
- Report only experiments that were actually executed. Clearly label incomplete components, mocks, and preliminary results.

Optional embeddings and mutation testing should be integrated when their corresponding components are ready. Do not treat a successful installation as evidence that those components are implemented or validated.

## 13. Quick Setup Checklist

Before beginning development, confirm:

- [ ] Python 3.12 is selected.
- [ ] `myvenv` is active.
- [ ] `requirements-lock.txt` has been used to install dependencies.
- [ ] The project is installed in editable mode.
- [ ] `pip check` succeeds.
- [ ] pytest and Ruff are available.
- [ ] Git is installed.
- [ ] Docker is available for sandbox work.
- [ ] API credentials are configured locally and are not committed.
- [ ] The test suite has been run.
- [ ] Work is happening on the appropriate feature branch.

**Team rule:** Keep the main branch reproducible, commit dependency changes deliberately, run tests before integration, and never report unexecuted experiments as results.

