# Scivianna Graph Plotter Module

The `plotter_graph` module provides the rendering backend for graph panels: it
displays a set of 2D plots (nodes) connected by directed links, typically used
to visualize the topology of coupled simulations.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization file |
| `generic_plotter.py` | Abstract base class `PlotterGraph` defining the graph rendering interface |
| `reactflow_plotter.py` | Reactflow-based plotter used by the graph panel for node preview (work in progress) |
