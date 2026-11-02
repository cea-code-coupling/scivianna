"""SVG icons used by the scivianna GUI."""

from pathlib import Path


def get_icon(icon_name: str):
    """Returns the content of the SVG icon with the given name.

    Parameters
    ----------
    icon_name : str
        Name of the icon (without the .svg extension)
    """
    with open(str(Path(__file__).parent / icon_name) + ".svg", "r") as f:
        return f.read()
