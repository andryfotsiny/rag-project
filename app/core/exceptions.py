
from typing import Any, Optional


class RAGBaseException(Exception):


    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class EmbeddingError(RAGBaseException):

    pass


class VectorStoreError(RAGBaseException):
    pass


class ChunkingError(RAGBaseException):
    """Erreur lors du découpage."""
    pass


class DocumentLoadError(RAGBaseException):
    """Erreur lors du chargement de document."""
    pass


class InsufficientResultsError(RAGBaseException):
    """Pas assez de résultats au-dessus du seuil."""
    pass


class ConfigurationError(RAGBaseException):
    """Erreur de configuration."""
    pass