"""FastAPI app: health and version endpoints."""

from fastapi import FastAPI
from pydantic import BaseModel

from retinaguard import DISCLAIMER, __version__

app = FastAPI(title="RetinaGuard API", version=__version__, description=DISCLAIMER)


class Health(BaseModel):
    status: str
    version: str
    models_loaded: bool


@app.get("/health")
def health() -> Health:
    # No models exist yet, so the service is up but cannot predict.
    return Health(status="ok", version=__version__, models_loaded=False)
