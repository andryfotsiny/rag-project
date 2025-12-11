from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_rag_pipeline_dep
from app.core.exceptions import RAGBaseException
from app.models.schemas import RAGRequest, RAGResponse
from app.services.rag_pipeline import RAGPipeline

router = APIRouter()


@router.post(
    "/rag",
    response_model=RAGResponse,
    summary="Pipeline RAG complet",
    description="Recherche les chunks pertinents, filtre par score, puis agrège le contexte final avec sources et métadonnées."
)
async def rag_query(
    request: RAGRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline_dep)
):
    try:
        result = await pipeline.process_query(
            query=request.query,
            k=request.k,
            min_score=request.min_score
        )
        
        return RAGResponse(**result)
        
    except RAGBaseException as e:
        raise HTTPException(status_code=500, detail=e.message)