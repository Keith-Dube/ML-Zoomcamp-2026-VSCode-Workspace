$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not installed. Install it from https://docs.astral.sh/uv/ and rerun this script."
}

uv python install 3.11
uv sync --group dev
uv run python -c "import numpy, pandas, sklearn; print('Environment ready:', numpy.__version__, pandas.__version__, sklearn.__version__)"

Write-Host ""
Write-Host "Setup complete. In VS Code, select: .venv\Scripts\python.exe"
Write-Host "Start here: modules\01-intro\README.md"

