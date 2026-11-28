import logging

from fastapi import FastAPI

from app.config import get_settings
from app.errors import AppError, app_error_handler
from app.routes import documents, health

logging.basicConfig(level=get_settings().log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per Ollama request; app.services.llm already logs each call

app = FastAPI(title="DocNotes", version="0.1.0")
app.add_exception_handler(AppError, app_error_handler)
app.include_router(health.router)
app.include_router(documents.router)
