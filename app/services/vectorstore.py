"""Service de gestion du vector store FAISS."""
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from loguru import logger

from app.core.config import settings
from app.core.constants import RelevanceLevel, RELEVANCE_THRESHOLDS
from app.core.exceptions import VectorStoreError
from app.models.domain import Chunk, SearchResult


class VectorStoreService:
    """Service pour gérer l'index FAISS et les recherches vectorielles."""

    def __init__(self):
        """Initialise le vector store."""
        self._index: Optional[faiss.Index] = None
        self._chunks: list[Chunk] = []
        self._dimension = settings.embedding_dimension
        self._index_path = settings.faiss_index_full_path
        self._metadata_path = settings.faiss_metadata_full_path

    def create_index(self) -> None:
        """Crée un nouvel index FAISS vide."""
        try:
            # IndexFlatIP = Inner Product (cosine similarity si normalisé)
            self._index = faiss.IndexFlatIP(self._dimension)
            logger.info(f"✅ Index FAISS créé (dimension: {self._dimension})")
        except Exception as e:
            raise VectorStoreError(
                "Impossible de créer l'index FAISS",
                details={"error": str(e)}
            )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """
        Ajoute des chunks à l'index.

        Args:
            chunks: Liste de chunks avec embeddings
        """
        if self._index is None:
            self.create_index()

        # Filtre les chunks avec embeddings
        valid_chunks = [c for c in chunks if c.embedding is not None]

        if not valid_chunks:
            raise VectorStoreError("Aucun chunk avec embedding à ajouter")

        try:
            # Convertit en numpy array
            embeddings = np.array([c.embedding for c in valid_chunks], dtype=np.float32)

            # Normalise pour cosine similarity
            if settings.normalize_embeddings:
                faiss.normalize_L2(embeddings)

            # Ajoute à l'index
            self._index.add(embeddings)
            self._chunks.extend(valid_chunks)

            logger.info(f"✅ {len(valid_chunks)} chunks ajoutés (total: {len(self._chunks)})")

        except Exception as e:
            raise VectorStoreError(
                "Échec de l'ajout des chunks",
                details={"error": str(e), "chunk_count": len(valid_chunks)}
            )

    def search(
            self,
            query_embedding: list[float],
            k: int = 5,
            min_score: Optional[float] = None
    ) -> list[SearchResult]:
        """
        Recherche les k chunks les plus similaires.

        Args:
            query_embedding: Embedding de la requête
            k: Nombre de résultats
            min_score: Score minimum (optionnel)

        Returns:
            Liste de SearchResult triés par score
        """
        if self._index is None or len(self._chunks) == 0:
            raise VectorStoreError("Index vide ou non chargé")

        min_score = min_score or settings.min_similarity_score

        try:
            # Convertit en numpy
            query_vec = np.array([query_embedding], dtype=np.float32)

            # Normalise
            if settings.normalize_embeddings:
                faiss.normalize_L2(query_vec)

            # Recherche (k * 2 pour avoir de la marge après filtrage)
            search_k = min(k * 2, len(self._chunks))
            scores, indices = self._index.search(query_vec, search_k)

            # Crée les résultats
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # Pas de résultat
                    continue

                # Filtre par score
                if score < min_score:
                    continue

                chunk = self._chunks[idx]
                relevance = self._get_relevance_level(score)

                results.append(SearchResult(
                    chunk=chunk,
                    score=float(score),
                    relevance=relevance
                ))

            # Garde seulement top-k
            results = results[:k]

            logger.info(f"Recherche : {len(results)} résultats (score >= {min_score})")
            return results

        except Exception as e:
            raise VectorStoreError(
                "Échec de la recherche",
                details={"error": str(e)}
            )

    def _get_relevance_level(self, score: float) -> RelevanceLevel:
        """Détermine le niveau de pertinence selon le score."""
        if score >= RELEVANCE_THRESHOLDS[RelevanceLevel.HIGH]:
            return RelevanceLevel.HIGH
        elif score >= RELEVANCE_THRESHOLDS[RelevanceLevel.MEDIUM]:
            return RelevanceLevel.MEDIUM
        else:
            return RelevanceLevel.LOW

    def save(self) -> None:
        """Sauvegarde l'index et les métadonnées."""
        if self._index is None:
            raise VectorStoreError("Aucun index à sauvegarder")

        try:
            # Crée les répertoires
            self._index_path.parent.mkdir(parents=True, exist_ok=True)

            # Sauvegarde FAISS
            faiss.write_index(self._index, str(self._index_path))

            # Sauvegarde métadonnées
            with open(self._metadata_path, "wb") as f:
                pickle.dump(self._chunks, f)

            logger.info(f"Index sauvegardé : {self._index_path}")
            logger.info(f"Métadonnées sauvegardées : {self._metadata_path}")

        except Exception as e:
            raise VectorStoreError(
                "Échec de la sauvegarde",
                details={"error": str(e)}
            )

    def load(self) -> None:
        """Charge l'index et les métadonnées."""
        if not self._index_path.exists():
            raise VectorStoreError(
                f"Index introuvable : {self._index_path}",
                details={"path": str(self._index_path)}
            )

        if not self._metadata_path.exists():
            raise VectorStoreError(
                f"Métadonnées introuvables : {self._metadata_path}",
                details={"path": str(self._metadata_path)}
            )

        try:
            # Charge FAISS
            self._index = faiss.read_index(str(self._index_path))

            # Charge métadonnées
            with open(self._metadata_path, "rb") as f:
                self._chunks = pickle.load(f)

            logger.info(f"Index chargé : {len(self._chunks)} chunks")

        except Exception as e:
            raise VectorStoreError(
                "Échec du chargement",
                details={"error": str(e)}
            )

    def is_loaded(self) -> bool:
        """Vérifie si l'index est chargé."""
        return self._index is not None and len(self._chunks) > 0

    def get_total_chunks(self) -> int:

        return len(self._chunks)

    def clear(self) -> None:

        self._index = None
        self._chunks = []
        logger.info("Index vidé")


# Singleton global
_vectorstore: Optional[VectorStoreService] = None


def get_vectorstore() -> VectorStoreService:

    global _vectorstore
    if _vectorstore is None:
        _vectorstore = VectorStoreService()
    return _vectorstore