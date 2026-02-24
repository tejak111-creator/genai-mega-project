from fastapi import FastAPI
from app.core.config import settings 
from app.core.logging import setup_logging
from app.core.middleware import request_context_middleware
from app.core.routes_health import router as health_router
from app.core.routes_chat import router as chat_router
from app.core.routes_rag import router as rag_router
from app.core.routes_agent import router as agent_router

"""This is called an application factory.

Instead of creating app = FastAPI() globally, you wrap it in a function."""
def create_app() -> FastAPI:
    setup_logging()
    
    app = FastAPI(title=settings.app_name)

    """"
    For every HTTP request, run request_context_middleware before and after the endpoint.
    http" means:

Apply middleware to HTTP requests.Wrapper func around all endpoints"""
    app.middleware("http")(request_context_middleware)
    """Incoming request
    ↓
    request_context_middleware
    ↓
    Route handler
    Middleware runs for every request."""
    app.include_router(health_router)
    app.include_router(chat_router)
    """Register Routers.This attaches route modules. NOW: /health, /chat are active endpoints"""

    #ADD THE RAG ROUTER
    app.include_router(rag_router)

    app.include_router(agent_router)
    
    return app

app= create_app()