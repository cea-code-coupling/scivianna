import atexit
import os
from pathlib import Path


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
            print(f"Deleting file {path}")
            os.remove(path)

    atexit.register(delete_file)
