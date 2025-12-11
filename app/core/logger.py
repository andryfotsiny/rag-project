"""Configuration Loguru pour logging structuré."""
import sys
from pathlib import Path

from loguru import logger

from app.core.config import settings


def setup_logging() -> None:
    """Configure Loguru avec rotation et niveaux appropriés."""

    # Supprime le handler par défaut
    logger.remove()

    # ============ CONSOLE (développement) ============
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=settings.log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # ============ FICHIER (production) ============
    log_file = settings.log_dir / "app.log"
    logger.add(
        log_file,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        ),
        level="INFO",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )

    # ============ ERRORS SÉPARÉS ============
    error_file = settings.log_dir / "errors.log"
    logger.add(
        error_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="ERROR",
        rotation="5 MB",
        retention="30 days",
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )

    logger.info(f"Logging configuré - Niveau: {settings.log_level}")


def log_with_context(level: str, message: str, **context) -> None:
    """Log avec contexte structuré."""
    context_str = " | ".join(f"{k}={v}" for k, v in context.items())
    full_message = f"{message} | {context_str}" if context else message

    log_func = getattr(logger, level.lower())
    log_func(full_message)