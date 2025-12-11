"""Application FastAPI principale."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.endpoints import embed, evaluate, health, rag, search
from app.core.config import settings
from app.core.logger import setup_logging
from app.services.vectorstore import get_vectorstore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle de l'application."""
    # Startup
    logger.info("Démarrage de l'application RAG")
    setup_logging()

    # Tente de charger l'index existant
    try:
        vectorstore = get_vectorstore()
        vectorstore.load()
        logger.info(f"Index chargé : {vectorstore.get_total_chunks()} chunks")
    except Exception as e:
        logger.warning(f" Index non chargé : {e}")
        logger.info("Utilisez le script d'ingestion pour créer l'index")

    yield

    # Shutdown
    logger.info("Arrêt de l'application")


# Création de l'app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API RAG avec FAISS et sentence-transformers",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health.router, prefix=settings.api_prefix, tags=["Health"])
app.include_router(embed.router, prefix=settings.api_prefix, tags=["Embeddings"])
app.include_router(search.router, prefix=settings.api_prefix, tags=["Search"])
app.include_router(rag.router, prefix=settings.api_prefix, tags=["RAG"])
app.include_router(evaluate.router, prefix=settings.api_prefix, tags=["Evaluation"])


@app.get("/")
def root():
    """Page d'accueil."""
    return {
        "message": "RAG API",
        "version": settings.app_version,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health"
    }