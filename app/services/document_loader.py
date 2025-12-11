
from pathlib import Path

from loguru import logger
from pypdf import PdfReader

from app.core.constants import FileType, SUPPORTED_EXTENSIONS
from app.core.exceptions import DocumentLoadError


class DocumentLoaderService:

    def load_file(self, file_path: Path) -> tuple[str, FileType]:
        """
        Charge un fichier et retourne son contenu + type.

        Args:
            file_path: Chemin vers le fichier

        Returns:
            Tuple (contenu, type_fichier)

        Raises:
            DocumentLoadError: Si le chargement échoue
        """
        if not file_path.exists():
            raise DocumentLoadError(
                f"Fichier introuvable : {file_path}",
                details={"path": str(file_path)}
            )

        extension = file_path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            raise DocumentLoadError(
                f"Extension non supportée : {extension}",
                details={"supported": list(SUPPORTED_EXTENSIONS)}
            )

        try:
            if extension == ".txt":
                return self._load_txt(file_path), FileType.TXT
            elif extension == ".md":
                return self._load_md(file_path), FileType.MD
            elif extension == ".pdf":
                return self._load_pdf(file_path), FileType.PDF
            else:
                raise DocumentLoadError(f"Type non géré : {extension}")

        except DocumentLoadError:
            raise
        except Exception as e:
            logger.error(f"Erreur chargement {file_path} : {e}")
            raise DocumentLoadError(
                f"Échec du chargement de {file_path.name}",
                details={"error": str(e)}
            )

    def _load_txt(self, file_path: Path) -> str:

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f" TXT chargé : {file_path.name} ({len(content)} chars)")
        return content

    def _load_md(self, file_path: Path) -> str:

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f" MD chargé : {file_path.name} ({len(content)} chars)")
        return content

    def _load_pdf(self, file_path: Path) -> str:

        try:
            reader = PdfReader(file_path)
            pages = []

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    pages.append(text)

            content = "\n\n".join(pages)
            logger.info(f"PDF chargé : {file_path.name} ({len(reader.pages)} pages, {len(content)} chars)")
            return content

        except Exception as e:
            raise DocumentLoadError(
                f"Erreur lecture PDF : {file_path.name}",
                details={"error": str(e)}
            )

    def load_directory(self, directory: Path) -> list[tuple[Path, str, FileType]]:
        """
        Charge tous les fichiers supportés d'un répertoire.

        Returns:
            Liste de (path, contenu, type)
        """
        if not directory.exists() or not directory.is_dir():
            raise DocumentLoadError(
                f"Répertoire introuvable : {directory}",
                details={"path": str(directory)}
            )

        results = []

        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    content, file_type = self.load_file(file_path)
                    results.append((file_path, content, file_type))
                except DocumentLoadError as e:
                    logger.warning(f"Fichier ignoré : {file_path.name} - {e.message}")

        logger.info(f" {len(results)} fichiers chargés depuis {directory}")
        return results


def get_document_loader() -> DocumentLoaderService:

    return DocumentLoaderService()