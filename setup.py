# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

import pathlib

here = pathlib.Path(__file__).parent.resolve()


# Get the long description from the README file
def get_long_description():
    """Extract README content"""
    return (here / "src" / "scivianna" / "readme.md").read_text(encoding="utf-8")

def get_version():
    """Extract the package's version number from the ``VERSION`` file."""
    return (here / "src" / "scivianna" / "VERSION").read_text(encoding="utf-8").strip()

setup(
    name="scivianna",
    version=get_version(),
    description="Python generic module to visualize simulation geometries and results.",
    long_description=get_long_description(),
    author="CEA",
    maintainer="Thibault Moulignier",
    author_email="Thibault.Moulignier@cea.fr",
    package_dir={"": "src"},  # Optional
    packages=find_packages(where="src"),  # Required
    package_data={
        "scivianna": [
            "agent/*.py",
            "agent/*.md",
            "input_file/*",
            "icon/*.svg",
            "utils/*",
            "*.sh",
            "VERSION",
            "readme.md"
        ],
        "scivianna_example": [
            "c3po_coupling/*",
            "europe_grid/*",
            "image/*.png",
            "med/*",
            "mandelbrot/*",
            "*/README.md",
            "*.md"
        ]
    },
    keywords="visualization",
    python_requires=">=3.8, <4",
    install_requires=[
        "panel",
        "rasterio",
        "matplotlib",
        "numpy",
        "shapely",
        "jupyter_bokeh",
        "holoviews",
        "icoco~=2.0.0",
        "panel_material_ui",
        "geopandas<1.1",
        "dill"
    ],
    extras_require={
        "default": [],
        "agent": [
            "smolagents[openai]",
        ],
        "pyvista": [
            "pyvista",
        ],
    },
)
