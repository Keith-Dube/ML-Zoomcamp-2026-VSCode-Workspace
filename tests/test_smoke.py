from mlzoomcamp.paths import PROJECT_ROOT


def test_project_root_contains_pyproject() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").exists()

