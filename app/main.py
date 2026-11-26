from fastapi import FastAPI

from app.errors import AppError, app_error_handler
from app.routes import documents, health

app = FastAPI(title="DocNotes", version="0.1.0")
app.add_exception_handler(AppError, app_error_handler)
app.include_router(health.router)
app.include_router(documents.router)
