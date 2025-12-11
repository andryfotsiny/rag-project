from fastapi import APIRouter, Depends, HTTPException

from app.core.exceptions import RAGBaseException
from app.models.schemas import EvaluateRequest, EvaluateResponse
from app.services.evaluation import EvaluationService, get_evaluation_service

router = APIRouter()


@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    summary="Évalue les performances du système RAG",
    description="Calcule les métriques Recall@K, Precision@K et similarité moyenne pour évaluer la qualité de la recherche."
)
async def evaluate_rag(
    request: EvaluateRequest,
    evaluation_service: EvaluationService = Depends(get_evaluation_service)
):
    """
    Métriques calculées :
    - Recall@K : Proportion des sources pertinentes retrouvées
    - Precision@K : Proportion des sources retrouvées qui sont pertinentes
    - Avg Similarity : Score de similarité moyen
    """
    try:
        queries = [
            {
                "query": q.query,
                "expected_sources": q.expected_sources
            }
            for q in request.queries
        ]

        result = await evaluation_service.evaluate_queries(queries, k=request.k)

        return EvaluateResponse(**result)

    except RAGBaseException as e:
        raise HTTPException(status_code=500, detail=e.message)