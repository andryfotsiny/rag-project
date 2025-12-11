from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_embedding_service_dep
from app.core.exceptions import EmbeddingError
from app.models.schemas import EmbedRequest, EmbedResponse
from app.services.embeddings import EmbeddingService

router = APIRouter()


@router.post(
    "/embed",
    response_model=EmbedResponse,
    summary="Transforme un texte en vecteur numérique",
    description="Génère un embedding (représentation vectorielle) pour un texte donné, utilisé pour la recherche sémantique."
)
async def embed_text(
        request: EmbedRequest,
        embedding_service: EmbeddingService = Depends(get_embedding_service_dep)
):
    try:
        embedding = await embedding_service.embed_text(request.text)

        return EmbedResponse(
            text=request.text,
            embedding=embedding,
            dimension=embedding_service.dimension,
            model=embedding_service.model_name
        )

    except EmbeddingError as e:
        raise HTTPException(status_code=500, detail=e.message)