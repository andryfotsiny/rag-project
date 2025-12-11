from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.core.constants import FileType, RelevanceLevel


class ChunkMetadata(BaseModel):
    """Métadonnées d'un chunk."""
    source: str = Field(..., description="Nom du fichier source")
    chunk_index: int = Field(..., ge=0, description="Index du chunk dans le document")
    total_chunks: int = Field(..., ge=1, description="Nombre total de chunks")
    file_type: FileType = Field(..., description="Type de fichier")
    created_at: datetime = Field(default_factory=datetime.now, description="Date de création")
    char_count: int = Field(..., ge=0, description="Nombre de caractères")
    start_char: int = Field(..., ge=0, description="Position de début dans le document")
    end_char: int = Field(..., ge=0, description="Position de fin dans le document")


class Chunk(BaseModel):
    """Représentation d'un chunk de texte."""
    id: str = Field(..., description="ID unique du chunk")
    text: str = Field(..., min_length=1, description="Contenu textuel")
    embedding: Optional[list[float]] = Field(None, description="Vecteur d'embedding")
    metadata: ChunkMetadata = Field(..., description="Métadonnées du chunk")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc1_chunk_0",
                "text": "Ceci est un exemple de chunk...",
                "metadata": {
                    "source": "document.pdf",
                    "chunk_index": 0,
                    "total_chunks": 10,
                    "file_type": "pdf",
                    "char_count": 1250,
                    "start_char": 0,
                    "end_char": 1250
                }
            }
        }


class Document(BaseModel):
    """Représentation d'un document complet."""
    filename: str = Field(..., description="Nom du fichier")
    file_type: FileType = Field(..., description="Type de fichier")
    content: str = Field(..., description="Contenu brut")
    chunks: list[Chunk] = Field(default_factory=list, description="Chunks générés")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Métadonnées supplémentaires")

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


class SearchResult(BaseModel):
    """Résultat d'une recherche vectorielle."""
    chunk: Chunk = Field(..., description="Chunk trouvé")
    score: float = Field(..., ge=0.0, le=1.0, description="Score de similarité")
    relevance: RelevanceLevel = Field(..., description="Niveau de pertinence")

    class Config:
        json_schema_extra = {
            "example": {
                "chunk": {
                    "id": "doc1_chunk_5",
                    "text": "Le RAG permet de...",
                    "metadata": {
                        "source": "rag_guide.pdf",
                        "chunk_index": 5,
                        "total_chunks": 20,
                        "file_type": "pdf"
                    }
                },
                "score": 0.89,
                "relevance": "high"
            }
        }