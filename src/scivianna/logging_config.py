"""
Logging configuration for Scivianna.

This module provides a centralized logging setup for the entire Scivianna package.
It configures logging with appropriate formatters, handlers, and log levels.

Example
-------
>>> from scivianna.logging import get_logger
>>> logger = get_logger(__name__)
>>> logger.info("Initialization complete")
"""

import logging
import os
import sys
from typing import Optional

def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the specified name.

    This function ensures consistent logging configuration across all Scivianna modules.
    If the root logger for 'scivianna' doesn't exist, it creates one with a console handler.

    Parameters
    ----------
    name : str
        Logger name, typically __name__ of the calling module

    Returns
    -------
    logging.Logger
        Configured logger instance

    Example
    -------
    >>> logger = get_logger(__name__)
    >>> logger.debug("Debug message")
    >>> logger.info("Info message")
    >>> logger.warning("Warning message")
    >>> logger.error("Error message")
    """
    # Get the logger
    logger = logging.getLogger(name)

    # Configure root scivianna logger if not already done
    scivianna_logger = logging.getLogger("scivianna")
    if not scivianna_logger.handlers:
        scivianna_logger.setLevel(logging.INFO)

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)

        # Add handler to scivianna logger
        scivianna_logger.addHandler(handler)

    # Set level based on environment variable
    log_level = int(getattr(logging, os.environ.get("SCIVIANNA_LOG_LEVEL", "INFO")))
    logger.setLevel(log_level)

    return logger


def set_log_level(level: int | str) -> None:
    """
    Set the log level for all Scivianna loggers.

    Parameters
    ----------
    level : int or str
        Logging level (e.g., logging.DEBUG, "DEBUG", 10)

    Example
    -------
    >>> set_log_level(logging.DEBUG)
    >>> set_log_level("DEBUG")
    """
    scivianna_logger = logging.getLogger("scivianna")
    if isinstance(level, str):
        level = int(getattr(logging, (level)))
    scivianna_logger.setLevel(level)

    # Update all handlers
    for handler in scivianna_logger.handlers:
        handler.setLevel(level)
