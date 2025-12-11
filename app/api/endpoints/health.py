from fastapi import APIRouter, Depends

from app.api.dependencies import get_vectorstore_dep
from app.core.config import settings
from app.models.schemas import HealthResponse
from app.services.vectorstore import VectorStoreService

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(vectorstore: VectorStoreService = Depends(get_vectorstore_dep)):

    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        vector_store_loaded=vectorstore.is_loaded(),
        total_chunks=vectorstore.get_total_chunks()
    )