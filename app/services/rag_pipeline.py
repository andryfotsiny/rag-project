
from typing import Optional

from loguru import logger

from app.core.config import settings
from app.core.exceptions import InsufficientResultsError
from app.models.domain import SearchResult
from app.services.embeddings import get_embedding_service
from app.services.vectorstore import get_vectorstore


class RAGPipeline:


    def __init__(self):

        self.embedding_service = get_embedding_service()
        self.vectorstore = get_vectorstore()

    async def process_query(
            self,
            query: str,
            k: int = 5,
            min_score: Optional[float] = None
    ) -> dict:

        logger.info(f"RAG Query : '{query}' (k={k})")

        # 1. Génère l'embedding de la query
        query_embedding = await self.embedding_service.embed_text(query)

        # 2. Recherche dans le vector store
        results = self.vectorstore.search(
            query_embedding=query_embedding,
            k=k,
            min_score=min_score
        )

        if not results:
            raise InsufficientResultsError(
                "Aucun résultat au-dessus du seuil",
                details={"min_score": min_score or settings.min_similarity_score}
            )


        context = self._aggregate_context(results)
        sources = self._extract_sources(results)
        scores = [r.score for r in results]


        metadata = {
            "chunk_count": len(results),
            "total_chars": len(context),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "sources_count": len(set(sources))
        }

        logger.info(f"✅ RAG : {len(results)} chunks, {len(context)} chars")

        return {
            "query": query,
            "context": context,
            "sources": sources,
            "scores": scores,
            "chunk_count": len(results),
            "metadata": metadata
        }

    def _aggregate_context(self, results: list[SearchResult]) -> str:
        """
        Agrège les chunks en contexte cohérent.

        Stratégie :
        - Concatène les chunks par ordre de score
        - Ajoute séparateurs entre chunks
        - Limite à max_context_length
        """
        chunks_text = []
        total_length = 0
        max_length = settings.max_context_length

        for result in results:
            chunk_text = result.chunk.text


            if total_length + len(chunk_text) > max_length:

                remaining = max_length - total_length
                if remaining > 100:
                    chunk_text = chunk_text[:remaining] + "..."
                    chunks_text.append(chunk_text)
                break

            chunks_text.append(chunk_text)
            total_length += len(chunk_text)

        # Concatène avec séparateurs
        context = "\n\n---\n\n".join(chunks_text)
        return context

    def _extract_sources(self, results: list[SearchResult]) -> list[str]:

        sources = []
        seen = set()

        for result in results:
            source = result.chunk.metadata.source
            if source not in seen:
                sources.append(source)
                seen.add(source)

        return sources


def get_rag_pipeline() -> RAGPipeline:

    return RAGPipeline()