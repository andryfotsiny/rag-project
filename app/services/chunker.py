from datetime import datetime

from loguru import logger

from app.core.config import settings
from app.core.constants import SEPARATOR_TOKEN, FileType
from app.core.exceptions import ChunkingError
from app.models.domain import Chunk, ChunkMetadata


class ChunkerService:

    def __init__(self):

        self._chunk_size = settings.chunk_size
        self._overlap = settings.chunk_overlap
        self._min_size = settings.min_chunk_size
        self._max_size = settings.max_chunk_size

    def chunk_text(
        self,
        text: str,
        source: str,
        file_type: FileType,
        chunk_size: int | None = None,
        overlap: int | None = None
    ) -> list[Chunk]:

        if not text or not text.strip():
            raise ChunkingError("Le texte ne peut pas être vide")

        chunk_size = chunk_size or self._chunk_size
        overlap = overlap or self._overlap

        if overlap >= chunk_size:
            raise ChunkingError(
                "L'overlap doit être inférieur à la taille du chunk",
                details={"chunk_size": chunk_size, "overlap": overlap}
            )

        try:
            logger.info(f"Découpage de '{source}' (taille: {chunk_size}, overlap: {overlap})")

            # 1 token ≈ 4 caractères en français
            char_chunk_size = chunk_size * 4
            char_overlap = overlap * 4

            chunks = self._smart_chunk_by_chars(text, char_chunk_size, char_overlap)

            # Crée les objets Chunk
            chunk_objects = []
            total_chunks = len(chunks)
            current_pos = 0

            for idx, chunk_text in enumerate(chunks):
                chunk_id = f"{source}_chunk_{idx}"
                char_count = len(chunk_text)

                metadata = ChunkMetadata(
                    source=source,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    file_type=file_type,
                    created_at=datetime.now(),
                    char_count=char_count,
                    start_char=current_pos,
                    end_char=current_pos + char_count
                )

                chunk_obj = Chunk(
                    id=chunk_id,
                    text=chunk_text.strip(),
                    metadata=metadata
                )

                chunk_objects.append(chunk_obj)
                current_pos += char_count

            logger.info(f"{len(chunk_objects)} chunks créés pour '{source}'")
            return chunk_objects

        except Exception as e:
            logger.error(f"Erreur découpage : {e}")
            raise ChunkingError(
                f"Échec du découpage de {source}",
                details={"error": str(e)}
            )

    def _smart_chunk_by_chars(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """
        Découpe par caractères avec overlap (RAPIDE, pas de fuite mémoire).

        Stratégie :
        - Découpe par paragraphes d'abord
        - Agrège jusqu'à atteindre chunk_size
        - Overlap entre chunks
        """
        # Sépare par paragraphes
        paragraphs = text.split(SEPARATOR_TOKEN)

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Si ajouter ce paragraphe dépasse la taille
            if len(current_chunk) + len(para) + 2 > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Garde les derniers X caractères pour overlap
                    current_chunk = current_chunk[-overlap:] + "\n\n" + para
                else:
                    # Paragraphe seul trop long, découpe brutalement
                    if len(para) > chunk_size:
                        for i in range(0, len(para), chunk_size - overlap):
                            chunks.append(para[i:i + chunk_size])
                    else:
                        current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Dernier chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def get_stats(self, text: str) -> dict[str, int]:
        chars = len(text)
        # Approximation : 1 token ≈ 4 chars
        estimated_tokens = chars // 4
        estimated_chunks = max(1, estimated_tokens // self._chunk_size)

        return {
            "chars": chars,
            "estimated_tokens": estimated_tokens,
            "estimated_chunks": estimated_chunks,
            "chunk_size": self._chunk_size,
            "overlap": self._overlap
        }


# Singleton global
_chunker_service: ChunkerService | None = None


def get_chunker_service() -> ChunkerService:
    """Retourne l'instance singleton du chunker."""
    global _chunker_service
    if _chunker_service is None:
        _chunker_service = ChunkerService()
    return _chunker_service
