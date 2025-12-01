from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from core.rag import rag_answer   # we will modify rag_answer to stream

router = APIRouter()


class Query(BaseModel):
    question: str


@router.post("/chat")
def chat_with_doc(payload: Query):

    # generator wrapper for StreamingResponse
    def stream():
        for chunk in rag_answer(payload.question):   # rag_answer MUST yield chunks
            yield chunk

    return StreamingResponse(stream(), media_type="text/plain")
