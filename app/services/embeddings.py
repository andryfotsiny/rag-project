
import asyncio
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.exceptions import EmbeddingError


class EmbeddingService:


    def __init__(self):

        self._model: SentenceTransformer | None = None
        self._model_name = settings.embedding_model
        self._device = settings.embedding_device
        self._batch_size = settings.embedding_batch_size
        self._normalize = settings.normalize_embeddings

    def _load_model(self) -> SentenceTransformer:

        if self._model is None:
            try:
                logger.info(f"Chargement du modèle d'embedding : {self._model_name}")
                self._model = SentenceTransformer(self._model_name, device=self._device)
                logger.info(f"Modèle chargé sur : {self._device}")
            except Exception as e:
                logger.error(f"Erreur chargement modèle : {e}")
                raise EmbeddingError(
                    f"Impossible de charger le modèle {self._model_name}",
                    details={"error": str(e)}
                )
        return self._model

    async def embed_text(self, text: str) -> list[float]:
        """
        Génère l'embedding d'un texte unique.

        Args:
            text: Texte à embedder

        Returns:
            Vecteur d'embedding

        Raises:
            EmbeddingError: Si la génération échoue
        """
        if not text or not text.strip():
            raise EmbeddingError("Le texte ne peut pas être vide")

        try:

            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self._generate_single_embedding,
                text
            )
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Erreur génération embedding : {e}")
            raise EmbeddingError(
                "Échec de la génération d'embedding",
                details={"text_length": len(text), "error": str(e)}
            )

    def _generate_single_embedding(self, text: str) -> np.ndarray:
        """Génère un embedding (synchrone)."""
        model = self._load_model()
        embedding = model.encode(
            text,
            normalize_embeddings=self._normalize,
            show_progress_bar=False
        )
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Génère les embeddings d'un batch de textes.

        Args:
            texts: Liste de textes à embedder

        Returns:
            Liste de vecteurs d'embedding

        Raises:
            EmbeddingError: Si la génération échoue
        """
        if not texts:
            raise EmbeddingError("La liste de textes ne peut pas être vide")

        valid_texts = [t for t in texts if t and t.strip()]
        if len(valid_texts) != len(texts):
            logger.warning(f"Textes vides ignorés : {len(texts) - len(valid_texts)}")

        if not valid_texts:
            raise EmbeddingError("Aucun texte valide à embedder")

        try:
            logger.info(f"Génération de {len(valid_texts)} embeddings...")


            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self._generate_batch_embeddings,
                valid_texts
            )

            logger.info(f"✅ {len(embeddings)} embeddings générés")
            return [emb.tolist() for emb in embeddings]

        except Exception as e:
            logger.error(f"Erreur génération batch : {e}")
            raise EmbeddingError(
                "Échec de la génération batch",
                details={"batch_size": len(valid_texts), "error": str(e)}
            )

    def _generate_batch_embeddings(self, texts: list[str]) -> np.ndarray:

        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=True
        )
        return embeddings

    @property
    def dimension(self) -> int:
        """Retourne la dimension des embeddings."""
        return settings.embedding_dimension

    @property
    def model_name(self) -> str:
        """Retourne le nom du modèle."""
        return self._model_name

    def is_loaded(self) -> bool:
        """Vérifie si le modèle est chargé."""
        return self._model is not None


# Singleton global
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Retourne l'instance singleton du service d'embedding."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service