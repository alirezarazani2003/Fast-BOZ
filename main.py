from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import g4f
import logging
import sys
import asyncio

g4f.check_version = False
g4f.debug.logging = True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fast BOZ",
    description="Free AI chat API using g4f. No API key required.",
    version="1.4.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []
    model: Optional[str] = "gpt-4o-mini"

class ChatResponse(BaseModel):
    response: str
    model: str
    provider: str

class ListModelsResponse(BaseModel):
    models: List[str]

class InfoResponse(BaseModel):
    service: str
    version: str
    g4f_version: str
    endpoints: dict


@app.get("/")
def health_check():
    return {"status": "healthy", "service": "AI Chat Service", "version": app.version,"message":"go to /docs or /info for more details <3"}


@app.get("/api/models", response_model=ListModelsResponse)
def list_models():
    try:
        available = [attr for attr in dir(g4f.models) if not attr.startswith("_")]
        return {"models": available}
    except Exception:
        raise HTTPException(status_code=500, detail="Could not retrieve models")


@app.get("/api/info", response_model=InfoResponse)
def info():
    return {
        "service": "AI Chat Service",
        "version": app.version,
        "g4f_version": getattr(g4f, "__version__", "unknown"),
        "endpoints": {
            "chat": "/api/chat (POST)",
            "models": "/api/models (GET)",
            "info": "/api/info (GET)",
            "docs": "/docs",
            "health": "/"
        }
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    logger.info(f"Chat request | Model: {request.model}")

    for msg in request.history:
        if msg.role not in ["user", "assistant"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid role. Only 'user' and 'assistant' are allowed."
            )

    try:
        messages = [{"role": m.role, "content": m.content} for m in request.history]
        messages.append({"role": "user", "content": request.message})

        try:
            model_name = request.model.replace("-", "_")
            model = getattr(g4f.models, model_name)
        except AttributeError:
            available = [m for m in dir(g4f.models) if not m.startswith("_")]
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' not found. Available: {', '.join(available)}"
            )

        try:
            response = await asyncio.wait_for(
                g4f.ChatCompletion.create_async(
                    model=model,
                    messages=messages,
                ),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="AI response timed out")

        if not response or not response.strip():
            raise HTTPException(status_code=502, detail="Empty response from AI")

        used_model = str(model).split(".")[-1].replace("_", "-")
        used_provider = "Unknown"
        if hasattr(g4f.ChatCompletion, "provider"):
            try:
                used_provider = g4f.ChatCompletion.provider.__name__
            except:
                used_provider = "Auto"

        clean_response = response.strip()
        logger.info(f"AI responded | Provider: {used_provider}, Length: {len(clean_response)}")

        return ChatResponse(
            response=clean_response,
            model=used_model,
            provider=used_provider
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI Error: {type(e).__name__}: {e}")
        if "Cloudflare" in str(e):
            raise HTTPException(
                status_code=503,
                detail="AI service is temporarily blocked"
            )
        raise HTTPException(status_code=500, detail="AI processing failed")