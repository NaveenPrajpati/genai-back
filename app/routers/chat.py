from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import storage

router = APIRouter(
    prefix="/chat", tags=["chat"], responses={404: {"description": "Not found"}}
)


class CreateChatRequest(BaseModel):
    title: Optional[str] = None


@router.post("", status_code=201)
async def create_chat_route(body: CreateChatRequest):
    """Create a new chat session. Title defaults to 'New Chat'."""
    try:
        chat_id = storage.create_chat(body.title or "New Chat")
        return {"message": "chat created", "data": storage.get_chat(chat_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", status_code=200)
async def get_all_chats():
    """Return all chats ordered by most recently updated."""
    try:
        return {"message": "chats fetched", "data": storage.list_chats()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{chat_id}/messages", status_code=200)
async def get_messages(chat_id: str):
    """Return a chat plus all its messages in chronological order."""
    try:
        return {
            "message": "messages fetched",
            "data": {
                "chat": storage.get_chat(chat_id),
                "messages": storage.get_messages(chat_id),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{chat_id}", status_code=200)
async def delete_chat(chat_id: str):
    """Delete a chat and all its messages (cascade)."""
    try:
        storage.delete_chat(chat_id)
        return {"message": "chat deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
