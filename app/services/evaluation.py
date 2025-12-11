
from loguru import logger

from app.services.embeddings import get_embedding_service
from app.services.vectorstore import get_vectorstore


class EvaluationService:


    def __init__(self):

        self.embedding_service = get_embedding_service()
        self.vectorstore = get_vectorstore()

    async def evaluate_queries(
        self,
        queries: list[dict],
        k: int = 5
    ) -> dict:
        """
        Évalue une liste de queries avec leurs sources attendues.

        Args:
            queries: Liste de {"query": str, "expected_sources": list[str]}
            k: Nombre de résultats à récupérer

        Returns:
            Dict avec métriques globales et détails par query
        """
        logger.info(f"Évaluation de {len(queries)} queries avec k={k}")

        total_recall = 0.0
        total_precision = 0.0
        total_similarity = 0.0
        details = []

        for query_data in queries:
            query = query_data["query"]
            expected_sources = set(query_data["expected_sources"])

            # Génère embedding et recherche
            query_embedding = await self.embedding_service.embed_text(query)
            results = self.vectorstore.search(
                query_embedding=query_embedding,
                k=k,
                min_score=0.0  # Pas de filtrage pour évaluation
            )

            found_sources = set([r.chunk.metadata.source for r in results])


            recall = self._calculate_recall(expected_sources, found_sources)
            precision = self._calculate_precision(expected_sources, found_sources)
            avg_score = sum(r.score for r in results) / len(results) if results else 0.0

            total_recall += recall
            total_precision += precision
            total_similarity += avg_score

            details.append({
                "query": query,
                "recall": round(recall, 3),
                "precision": round(precision, 3),
                "avg_score": round(avg_score, 3),
                "found_sources": sorted(list(found_sources)),
                "expected_sources": sorted(list(expected_sources)),
                "retrieved_count": len(results)
            })

        # Moyennes globales
        n = len(queries)
        return {
            "recall_at_k": round(total_recall / n, 3),
            "precision_at_k": round(total_precision / n, 3),
            "avg_similarity": round(total_similarity / n, 3),
            "total_queries": n,
            "details": details
        }

    def _calculate_recall(self, expected: set, found: set) -> float:

        if not expected:
            return 1.0

        intersection = expected.intersection(found)
        return len(intersection) / len(expected)

    def _calculate_precision(self, expected: set, found: set) -> float:
        if not found:
            return 0.0  # Si rien trouvé, zéro

        intersection = expected.intersection(found)
        return len(intersection) / len(found)


def get_evaluation_service() -> EvaluationService:
    return EvaluationService()
