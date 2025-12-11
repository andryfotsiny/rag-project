from typing import Any, Optional
from pydantic import BaseModel, Field
from app.models.domain import SearchResult


class EmbedRequest(BaseModel):
    """Request pour générer un embedding."""
    text: str = Field(..., min_length=1, max_length=10000, description="Texte à embedder")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Comment fonctionne le système RAG ?"
            }
        }


class EmbedResponse(BaseModel):
    """Response avec l'embedding généré."""
    text: str = Field(..., description="Texte original")
    embedding: list[float] = Field(..., description="Vecteur d'embedding")
    dimension: int = Field(..., description="Dimension du vecteur")
    model: str = Field(..., description="Modèle utilisé")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Comment fonctionne le système RAG ?",
                "embedding": [0.123, -0.456, 0.789, "..."],
                "dimension": 384,
                "model": "all-MiniLM-L6-v2"
            }
        }


class SearchRequest(BaseModel):
    """Request pour rechercher dans le vector store."""
    query: str = Field(..., min_length=1, max_length=1000, description="Requête de recherche")
    k: int = Field(5, ge=1, le=20, description="Nombre de résultats")
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Score minimum")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Comment fonctionne le système ?",
                "k": 5,
                "min_score": 0.35
            }
        }


class SearchResponse(BaseModel):
    """Response avec les résultats de recherche."""
    query: str = Field(..., description="Requête originale")
    results: list[SearchResult] = Field(..., description="Résultats trouvés")
    total_found: int = Field(..., description="Nombre total de résultats")
    filtered_count: int = Field(..., description="Nombre après filtrage")
    min_score_used: float = Field(..., description="Score minimum appliqué")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Comment fonctionne le système ?",
                "results": [
                    {
                        "chunk": {
                            "id": "doc1_chunk_5",
                            "text": "Le système fonctionne...",
                            "metadata": {"source": "guide.pdf"}
                        },
                        "score": 0.89,
                        "relevance": "high"
                    }
                ],
                "total_found": 8,
                "filtered_count": 5,
                "min_score_used": 0.65
            }
        }

class RAGRequest(BaseModel):
    """Request pour le pipeline RAG complet."""
    query: str = Field(..., min_length=1, max_length=1000, description="Question utilisateur")
    k: int = Field(5, ge=1, le=20, description="Nombre de chunks à récupérer")
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Score minimum")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Quels sont les avantages du RAG ?",
                "k": 5,
                "min_score": 0.35
            }
        }


class RAGResponse(BaseModel):
    """Response du pipeline RAG."""
    query: str = Field(..., description="Question originale")
    context: str = Field(..., description="Contexte agrégé des meilleurs chunks")
    sources: list[str] = Field(..., description="Sources des documents")
    scores: list[float] = Field(..., description="Scores de similarité")
    chunk_count: int = Field(..., description="Nombre de chunks utilisés")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Métadonnées supplémentaires")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Quels sont les avantages du RAG ?",
                "context": "Le RAG permet de combiner... Les avantages incluent...",
                "sources": ["rag_guide.pdf", "architecture.md"],
                "scores": [0.89, 0.85, 0.82],
                "chunk_count": 3,
                "metadata": {
                    "total_chars": 1500,
                    "avg_score": 0.85
                }
            }
        }


class HealthResponse(BaseModel):
    """Response du healthcheck."""
    status: str = Field(..., description="Status de l'API")
    version: str = Field(..., description="Version de l'API")
    vector_store_loaded: bool = Field(..., description="Vector store chargé")
    total_chunks: int = Field(0, description="Nombre de chunks indexés")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "vector_store_loaded": True,
                "total_chunks": 1250
            }
        }


class ErrorResponse(BaseModel):
    """Response d'erreur standard."""
    error: str = Field(..., description="Type d'erreur")
    message: str = Field(..., description="Message d'erreur")
    details: Optional[dict[str, Any]] = Field(None, description="Détails supplémentaires")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "VectorStoreError",
                "message": "Index FAISS non trouvé",
                "details": {
                    "path": "data/processed/index.faiss"
                }
            }
        }


# ============ EVALUATE ENDPOINT (BONUS) ============

class EvaluateQuery(BaseModel):
    """Une query d'évaluation avec sources attendues."""
    query: str = Field(..., description="Question à évaluer")
    expected_sources: list[str] = Field(..., description="Sources attendues")


class EvaluateRequest(BaseModel):
    """Request pour évaluer le système RAG."""
    queries: list[EvaluateQuery] = Field(..., min_length=1, description="Queries à évaluer")
    k: int = Field(5, ge=1, le=20, description="Nombre de résultats à récupérer")

    class Config:
        json_schema_extra = {
            "example": {
                "queries": [
                    {
                        "query": "Quels sont les avantages du RAG ?",
                        "expected_sources": ["intro.txt", "faq.txt"]
                    },
                    {
                        "query": "Comment fonctionne l'architecture ?",
                        "expected_sources": ["architecture.md"]
                    }
                ],
                "k": 5
            }
        }


class EvaluateResponse(BaseModel):
    """Response avec métriques d'évaluation."""
    recall_at_k: float = Field(..., description="Recall@K moyen")
    precision_at_k: float = Field(..., description="Précision@K moyenne")
    avg_similarity: float = Field(..., description="Similarité moyenne")
    total_queries: int = Field(..., description="Nombre de queries évaluées")
    details: list[dict[str, Any]] = Field(..., description="Détails par query")

    class Config:
        json_schema_extra = {
            "example": {
                "recall_at_k": 0.85,
                "precision_at_k": 0.60,
                "avg_similarity": 0.72,
                "total_queries": 2,
                "details": [
                    {
                        "query": "Quels sont les avantages du RAG ?",
                        "recall": 1.0,
                        "precision": 0.6,
                        "avg_score": 0.75,
                        "found_sources": ["intro.txt", "faq.txt", "architecture.md"],
                        "expected_sources": ["intro.txt", "faq.txt"]
                    }
                ]
            }
        }