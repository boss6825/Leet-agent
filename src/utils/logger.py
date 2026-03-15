"""Simple structured logger using rich."""
import logging
from rich.logging import RichHandler
from src.config import LOG_LEVEL


def setup_logger(name: str = "leetcode-agent") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )
    return logging.getLogger(name)


logger = setup_logger()
