"""
SHL Assessment Recommender — FastAPI Service
GET  /health  → readiness check
POST /chat    → stateless conversational agent
"""
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Agent singleton — initialized at startup
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        from agent import SHLAgent
        _agent = SHLAgent()
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up agent on startup to avoid cold-start latency on first request
    try:
        get_agent()
        logger.info("Agent warmed up successfully")
    except Exception as e:
        logger.error(f"Agent warmup failed: {e}")
    yield


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational AI agent for finding SHL Individual Test Solutions",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Models ─────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty")
        return v


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Readiness probe. Returns 200 + {"status": "ok"} when ready."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Stateless conversational endpoint.
    Send the FULL conversation history on every call.
    Max 8 messages (user + assistant turns combined) per conversation.
    """
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Enforce 8-turn cap required by evaluator
    if len(messages) > 8:
        raise HTTPException(
            status_code=400,
            detail=f"Conversation has {len(messages)} messages; max is 8 (user + assistant)."
        )

    try:
        result = get_agent().respond(messages)
    except ValueError as e:
        logger.error(f"Config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error. Please retry.")

    return ChatResponse(
        reply=result["reply"],
        recommendations=[Recommendation(**r) for r in result["recommendations"]],
        end_of_conversation=result["end_of_conversation"],
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Unexpected server error"})


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
