# Scivianna: SCIentific VIsualizer for simulAtioN aNAlysis

Scivianna is an open-source Python simulation geometry and result visualizer based on Holoviz Panel. It provides a powerful and flexible environment for visualizing 1D plots and 2D geometries simultaneously, with support for real-time simulation result visualization through code coupling.

## Installation

Scivianna can be installed directly from PyPI using pip:

```bash
pip install scivianna
```

### Optional Dependencies

Scivianna offers several optional dependency sets for specific features:

```bash
# Agent/LLM features
pip install scivianna[agent]

# PyVista 3D visualization
pip install scivianna[pyvista]

# Testing and development
pip install scivianna[test]

# Code coupling with C3PO
pip install scivianna[coupling]
```

## Quick Start

### Basic Usage

```python
from scivianna.layout.gridstack import GridStackLayout
from scivianna.panel.visualisation_panel import VisualizationPanel
from scivianna.slave import ComputeSlave

# Create a slave with your data interface
slave = ComputeSlave(YourInterface)

# Create a visualization panel
panel = VisualizationPanel(slave, name="My Panel")

# Create a layout and add the panel
layout = GridStackLayout({"panel1": panel})

# Display
layout.show()
```

### Running the Demonstrator

Scivianna includes a `scivianna_example` module with several demonstration cases. To run the demonstrator:

```python
from scivianna_example.demo import make_demo

demo = make_demo()
demo.show()
```

## scivianna_example Module

The `scivianna_example` module provides ready-to-use examples showcasing Scivianna's capabilities:

### Available Examples

| Example | Description |
|---------|-------------|
| **Europe Grid** | Plots an interactive Europe map displaying electricity production and consumption per country with weekly time-steps. Hover over countries to see values. |
| **Medcoupling** | Demonstrates MED file format handling with three synchronized views. Click in one view to offset others at the click location. |
| **Mandelbrot** | Computes the Mandelbrot set on a 2D grid and converts it to polygons for visualization. |

### Accessing Examples Individually

```python
# Europe example
from scivianna_example.europe_grid.europe_grid import make_europe_panel
europe_panel = make_europe_panel(None)

# Medcoupling example  
from scivianna_example.med.split_item_example import get_panel
medcoupling_panel = get_panel(None)

# Mandelbrot example
from scivianna_example.mandelbrot.mandelbrot import make_panel
mandelbrot_panel = make_panel(None)
```

## Repository Organization

```
scivianna/
├── src/
│   └── scivianna/              # Main package source code
│       ├── agent/              # LLM/Agent integration tools
│       ├── component/          # UI components (GridStack, Overlay, SplitJS, R3F)
│       ├── coupling/           # C3PO/ICOCO coupling interface
│       ├── data/               # Data containers and workers
│       ├── extension/          # Panel extensions (axes, layout, file loader, etc.)
│       ├── icon/               # UI icons
│       ├── input_file/         # Sample input files
│       ├── interface/          # Data interfaces (MED, VTK, CSV, generic)
│       ├── layout/             # Layout managers (GridStack, Split)
│       ├── panel/              # Visualization panels and GUI
│       ├── plotter_1d/         # 1D plotting backends
│       ├── plotter_2d/         # 2D plotting backends (grid, polygon)
│       └── utils/              # Utilities (mesh tools, serialization, etc.)
│
├── tests/                      # Test suite
├── utils/                      # Development utilities
└── results/                    # Example outputs
```

### Core Modules

| Module | Purpose |
|--------|---------|
| **panel/** | Visualization panels with GUI controls, extensions, and overlay components |
| **layout/** | Layout managers for arranging multiple panels (GridStack, Split) |
| **interface/** | Data interfaces for reading various file formats (MED, VTK, CSV) |
| **coupling/** | ICOCO-compatible interface for code coupling with C3PO |
| **data/** | Data containers and workers for handling simulation data |
| **extension/** | Extendable components for axes, field selection, file loading, etc. |
| **plotter_1d/** & **plotter_2d/** | Plotting backends using Bokeh and Matplotlib |
| **component/** | Custom Panel components including React Three Fiber integration |

## Key Features

### 1. Simultaneous 1D and 2D Visualization
Visualize line plots and 2D geometries together in an integrated dashboard.

### 2. Real-Time Code Coupling
Connect to simulations in real-time using the C3PO coupling platform and ICOCO interface. The visualizer can receive field updates during simulation execution.

### 3. Generic Data Interfaces
Scivianna provides flexible interfaces to read various data formats:
- **MED** (Salome MED file format)
- **VTK** (Visualization Toolkit files)
- **CSV** (Structured time-series data)
- **Custom interfaces** via `GenericInterface` base class

### 4. Remote Server Access
Access the visualizer from a distant computer using Panel's server/client architecture. Perfect for running simulations on HPC clusters while visualizing locally.

### 5. Extensible Architecture
Extensions can be added to panels for additional functionality:
- Field selectors
- Line selection tools
- File loaders
- Layout management
- AI assistant integration

## Code Coupling with C3PO

Scivianna integrates with the C3PO coupling tool through the ICOCO interface:

```python
from scivianna.coupling.icoco import LayoutProblem
from scivianna.layout.gridstack import GridStackLayout

# Create your layout
layout = GridStackLayout(panels)

# Create the coupling problem
problem = LayoutProblem(layout, title="Coupled Simulation")

# Use with C3PO
# problem.initialize()
# problem.setInputMEDDoubleField("panel_name@field_name", field_data)
```

## Requirements

- Python >= 3.8, < 4
- panel
- holoviews
- bokeh / jupyter_bokeh
- matplotlib
- numpy / pandas
- shapely / geopandas
- rasterio
- icoco (for coupling features)
- panel_material_ui

## Demo Application

A live demonstrator (excluding real-time simulation features) is available at:
https://huggingface.co/spaces/tmoulignier/scivianna

## License

Scivianna is developed by CEA (Commissariat à l'énergie atomique et aux énergies alternatives).

## Contact

- **Maintainer**: Thibault Moulignier
- **Email**: Thibault.Moulignier@cea.fr