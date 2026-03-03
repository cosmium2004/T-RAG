"""
Utility: Application-wide structured logger.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  Call ``setup_logging()`` first."""
    return logging.getLogger(name)
