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
pip install scivianna[3d]

# Testing and development
pip install scivianna[test]

# Code coupling with C3PO
pip install scivianna[coupling]
```

## Quick Start

### Basic Usage

```python
from scivianna.layout.gridstack import GridStackLayout
from scivianna.panel.panel2d import Panel2D
from scivianna.slave import ComputeSlave

# Create a slave with your data interface
slave = ComputeSlave(YourInterface)

# Create a visualization panel
panel = Panel2D(slave, name="My Panel")

# Display
panel.show()
```

### Running the Demonstrator
Scivianna includes a `scivianna_example` module with several demonstration cases. 

It can be opened online at:
https://huggingface.co/spaces/tmoulignier/scivianna

or by running:

```python
from scivianna_example.demo import make_demo

demo = make_demo()
demo.show()
```


## Key Features

### 1. Simultaneous 1D, 2D and 3D Visualization

Visualize line plots, 2D geometries and vtk.js plots together in an integrated dashboard with inter-plot interactions.

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

## Panel interactions

Visualisation panels can interact with one another based on a update event parameter. 


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

## scivianna_example Module

The `scivianna_example` module provides ready-to-use examples showcasing Scivianna's capabilities:

### Available Examples

| Example | Description |
|---------|-------------|
| **Europe Grid** | Plots an interactive Europe map displaying electricity production and consumption per country with weekly time-steps. Hover over countries to see values. |
| **Mandelbrot** | Computes the Mandelbrot set on a 2D grid and converts it to polygons for visualization. |
| **Medcoupling** | Demonstrates MED file format handling with three synchronized views. Click in one view to offset others at the click location. (if medcoupling  available) |
| **Coupling example** | Plot of a coupled simulation to see time dependant data (if salome-c3po installed) |
| **3D example** | 3D plot interacting with a 2D plot. (if medcoupling and pyvista available) |


## License

Scivianna is developed by CEA (Commissariat à l'énergie atomique et aux énergies alternatives).

## Contact

- **Maintainer**: Thibault Moulignier
- **Email**: Thibault.Moulignier@cea.fr
