# Machine Learning Zoomcamp 2026 Workspace

One VS Code workspace for the complete DataTalksClub Machine Learning Zoomcamp.

## Quick start on Windows

1. Install [VS Code](https://code.visualstudio.com/), [Git](https://git-scm.com/), and [uv](https://docs.astral.sh/uv/).
2. Open this folder in VS Code.
3. In PowerShell, run:

```powershell
.\scripts\setup.ps1
```

4. Select the `.venv` Python interpreter when VS Code prompts you.
5. Open `modules/01-intro/README.md` and begin with its checklist.

## Common commands

```powershell
uv sync --group dev
uv run jupyter lab
uv run pytest
uv run ruff check .
uv run ruff format .
```

Install only the heavier tools when their module begins:

```powershell
uv sync --extra trees
uv sync --extra deployment
uv sync --extra deep-learning
```

## Workspace map

```text
ML-Zoomcamp-2026/
├── .vscode/              VS Code settings, extensions, and tasks
├── data/                 Local raw/processed data (ignored by Git)
├── artifacts/            Models, figures, and reports (large files ignored)
├── docs/                 Progress and question tracking
├── modules/              Course modules in official sequence
├── projects/             Midterm and capstone workspaces
├── src/mlzoomcamp/       Reusable code shared across modules
├── templates/            Reusable notes and Python workbook templates
├── tests/                Shared smoke tests
├── pyproject.toml        uv project and dependency configuration
└── README.md             Start here
```

## Working rules

- Keep reusable logic in `src/mlzoomcamp`; keep exploration in module workbooks.
- Do not commit datasets, secrets, virtual environments, or large model files.
- Record the source and license for every dataset in `data/README.md`.
- Restart and run notebooks/workbooks top-to-bottom before marking work complete.
- Commit small, meaningful steps and keep each project reproducible.

## Official course

- [Repository](https://github.com/DataTalksClub/machine-learning-zoomcamp)
- [2026 cohort page](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/cohorts/2026)
