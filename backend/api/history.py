from fastapi import APIRouter, Body
from core.db import get_messages, save_message

router = APIRouter(prefix="/api/history", tags=["history"])

@router.get("/")
def fetch_history(project_path: str):
    messages = get_messages(project_path)
    return {"messages": messages}

@router.post("/")
def add_history(project_path: str = Body(...), message: dict = Body(...)):
    save_message(project_path, message)
    return {"status": "ok"}
