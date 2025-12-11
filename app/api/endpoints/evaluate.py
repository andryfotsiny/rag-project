from fastapi import APIRouter, Depends, HTTPException

from app.core.exceptions import RAGBaseException
from app.models.schemas import EvaluateRequest, EvaluateResponse
from app.services.evaluation import EvaluationService, get_evaluation_service

router = APIRouter()


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_rag(
    request: EvaluateRequest,
    evaluation_service: EvaluationService = Depends(get_evaluation_service)
):
    """
    Évalue la performance du système RAG.
    
    Métriques calculées :
    - **Recall@K** : Proportion des sources pertinentes retrouvées
    - **Precision@K** : Proportion des sources retrouvées qui sont pertinentes
    - **Avg Similarity** : Score de similarité moyen
    
    Exemple :
```json
    {
      "queries": [
        {
          "query": "Quels sont les avantages du RAG ?",
          "expected_sources": ["intro.txt", "faq.txt"]
        }
      ],
      "k": 5
    }
```
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
