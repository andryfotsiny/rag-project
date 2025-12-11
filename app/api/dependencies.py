
from app.services.embeddings import get_embedding_service
from app.services.rag_pipeline import get_rag_pipeline
from app.services.vectorstore import get_vectorstore


def get_embedding_service_dep():
    return get_embedding_service()

def get_vectorstore_dep():
    return get_vectorstore()


def get_rag_pipeline_dep():
    return get_rag_pipeline()