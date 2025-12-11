
from enum import Enum


class FileType(str, Enum):

    TXT = "txt"
    MD = "md"
    PDF = "pdf"


class RelevanceLevel(str, Enum):

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IndexType(str, Enum):
    """Types d'index FAISS."""
    FLAT_IP = "IndexFlatIP"
    FLAT_L2 = "IndexFlatL2"



# Mapping des seuils de pertinence
RELEVANCE_THRESHOLDS = {
    RelevanceLevel.HIGH: 0.80,
    RelevanceLevel.MEDIUM: 0.65,
    RelevanceLevel.LOW: 0.0,
}

# Extensions de fichiers supportées
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

# Tokens spéciaux
SEPARATOR_TOKEN = "\n\n"