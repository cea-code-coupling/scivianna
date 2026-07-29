"""
File cleaner utility for Scivianna.

This module provides utilities for marking files for deletion on program exit.
"""

import atexit
import os
from pathlib import Path

from scivianna.logging_config import get_logger

logger = get_logger(__name__)


def mark_for_deletion(path: Path):
    """Marks a file to be deleted when the visualizer is closed.

    Parameters
    ----------
    path : Path
        File path
    """

    def delete_file():
        """Deletes the file at the given path if it still exits"""
        if os.path.isfile(path):
            logger.debug("Deleting file: %s", path)
            os.remove(path)

    atexit.register(delete_file)
