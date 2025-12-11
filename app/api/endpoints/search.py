from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_embedding_service_dep, get_vectorstore_dep
from app.core.exceptions import EmbeddingError, VectorStoreError
from app.models.schemas import SearchRequest, SearchResponse
from app.services.embeddings import EmbeddingService
from app.services.vectorstore import VectorStoreService

router = APIRouter()


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Recherche les chunks similaires",
    description="Recherche dans la base vectorielle les k chunks les plus similaires à la requête selon le score de similarité."
)
async def search_vectors(
    request: SearchRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service_dep),
    vectorstore: VectorStoreService = Depends(get_vectorstore_dep)
):
    try:
        query_embedding = await embedding_service.embed_text(request.query)

        results = vectorstore.search(
            query_embedding=query_embedding,
            k=request.k,
            min_score=request.min_score
        )
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_found=len(results),
            filtered_count=len(results),
            min_score_used=request.min_score or 0.65
        )
        
    except (EmbeddingError, VectorStoreError) as e:
        raise HTTPException(status_code=500, detail=e.message)