from fastapi import FastAPI

from app.routes import chat, documents, health

app = FastAPI(title="DocNotes", version="0.1.0")
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)
