from fastapi import APIRouter, Depends

from app.api.dependencies import get_vectorstore_dep
from app.core.config import settings
from app.models.schemas import HealthResponse
from app.services.vectorstore import VectorStoreService

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérifie l'état du système",
    description="Retourne l'état de santé de l'API, la version, et le nombre de documents indexés dans la base vectorielle."
)
def health_check(vectorstore: VectorStoreService = Depends(get_vectorstore_dep)):
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        vector_store_loaded=vectorstore.is_loaded(),
        total_chunks=vectorstore.get_total_chunks()
    )