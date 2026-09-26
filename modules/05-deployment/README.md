# Module 5 — Deploying Machine Learning Models

Install the deployment dependencies:

```powershell
uv sync --extra deployment --group dev
uv run uvicorn service.app:app --app-dir modules/05-deployment --reload
uv run pytest modules/05-deployment/service/test_app.py
docker build -f modules/05-deployment/Dockerfile -t mlzoomcamp-api .
```

- [ ] Export the preprocessing pipeline and model together.
- [ ] Serve predictions with a typed API.
- [ ] Add a health endpoint and tests.
- [ ] Build and smoke-test the Docker image.

Official materials: https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/05-deployment
