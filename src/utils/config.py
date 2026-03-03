"""
Utility: Structured logging and configuration management for T-RAG.
"""

import os
import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
) -> None:
    """Configure structured logging for the application."""
    load_dotenv()
    level = os.getenv("LOG_LEVEL", level)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(fmt))
    root.addHandler(console)

    # File handler (optional)
    log_path = log_file or os.getenv("LOG_FILE")
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10_000_000, backupCount=5
        )
        fh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load YAML config with environment variable substitution.

    Variables in the form ``${VAR_NAME}`` are replaced with their
    environment-variable values.
    """
    load_dotenv()

    with open(config_path, "r") as f:
        raw = f.read()

    # Substitute ${VAR} patterns
    import re
    pattern = re.compile(r"\$\{(\w+)\}")
    def _replace(match):
        return os.getenv(match.group(1), match.group(0))

    resolved = pattern.sub(_replace, raw)
    return yaml.safe_load(resolved)
