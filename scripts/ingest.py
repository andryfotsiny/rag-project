"""Script d'ingestion de documents dans le vector store."""
import asyncio
import sys
from pathlib import Path

from loguru import logger

# Ajoute le path du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.logger import setup_logging
from app.services.chunker import get_chunker_service
from app.services.document_loader import get_document_loader
from app.services.embeddings import get_embedding_service
from app.services.vectorstore import get_vectorstore


async def ingest_documents(data_dir: Path | None = None):
    """
    Ingère tous les documents d'un répertoire.

    Args:
        data_dir: Répertoire contenant les documents (par défaut: data/raw)
    """
    # Setup
    setup_logging()
    logger.info("=" * 60)
    logger.info("DÉBUT DE L'INGESTION")
    logger.info("=" * 60)

    data_dir = data_dir or settings.data_dir

    # Services
    loader = get_document_loader()
    chunker = get_chunker_service()
    embedding_service = get_embedding_service()
    vectorstore = get_vectorstore()

    # 1. Charge les documents
    logger.info(f"Chargement depuis : {data_dir}")
    documents = loader.load_directory(data_dir)

    if not documents:
        logger.error(f"❌ Aucun document trouvé dans {data_dir}")
        return

    logger.info(f"{len(documents)} documents chargés")

    # 2. Découpe en chunks
    logger.info("🔪 Découpage en chunks...")
    all_chunks = []

    for file_path, content, file_type in documents:
        try:
            chunks = chunker.chunk_text(
                text=content,
                source=file_path.name,
                file_type=file_type
            )
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"❌ Erreur découpage {file_path.name} : {e}")

    if not all_chunks:
        logger.error("❌ Aucun chunk créé")
        return

    logger.info(f"{len(all_chunks)} chunks créés")

    # 3. Génère les embeddings
    logger.info("Génération des embeddings...")
    texts = [chunk.text for chunk in all_chunks]

    try:
        embeddings = await embedding_service.embed_batch(texts)

        # Attache les embeddings aux chunks
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk.embedding = embedding

        logger.info(f"{len(embeddings)} embeddings générés")

    except Exception as e:
        logger.error(f"❌ Erreur génération embeddings : {e}")
        return

    # 4. Indexation dans FAISS
    logger.info("Indexation dans FAISS...")

    try:
        vectorstore.create_index()
        vectorstore.add_chunks(all_chunks)

        logger.info(f"{len(all_chunks)} chunks indexés")

    except Exception as e:
        logger.error(f"❌ Erreur indexation : {e}")
        return

    # 5. Sauvegarde
    logger.info("Sauvegarde de l'index...")

    try:
        vectorstore.save()
        logger.info("Index sauvegardé")

    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde : {e}")
        return

    # Résumé
    logger.info("=" * 60)
    logger.info("INGESTION TERMINÉE AVEC SUCCÈS")
    logger.info(f"📊 Documents : {len(documents)}")
    logger.info(f"📊 Chunks : {len(all_chunks)}")
    logger.info(f"📊 Index : {settings.faiss_index_full_path}")
    logger.info("=" * 60)


def main():
    """Point d'entrée du script."""
    # Parse arguments (optionnel)
    import argparse

    parser = argparse.ArgumentParser(description="Ingestion de documents")
    parser.add_argument(
        "--data-dir",
        type=str,
        help=f"Répertoire des documents (défaut: {settings.data_dir})"
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else None

    # Lance l'ingestion
    asyncio.run(ingest_documents(data_dir))


if __name__ == "__main__":
    main()