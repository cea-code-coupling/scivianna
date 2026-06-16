# Scivianna Example Package

This package provides a collection of demonstration examples showcasing Scivianna's visualization capabilities. The examples are designed to help users understand how to create interactive visualizations, work with different data formats, and build complex multi-view layouts.

## Overview

The `scivianna_example` package contains several use cases:

| Example | Description |
|---------|-------------|
| **Europe Grid** | Visualize electricity production and consumption across European countries using GeoJSON maps |
| **MED Coupling** | Display Salome MED file data with synchronized multi-view displays |
| **Mandelbrot** | Compute and visualize the Mandelbrot fractal set using polygon-based rendering |
| **C3PO Coupling** | Integrate Scivianna with the C3PO coupling platform for real-time simulation |

## Installation

Ensure Scivianna is installed:

```bash
pip install scivianna
```

## Quick Start

Run the main demonstrator to explore all examples:

```python
from scivianna_example.demo import make_demo

# Launch the interactive demonstrator
demo = make_demo()
demo.show()
```

The demonstrator provides a navigation menu to switch between different examples.

## Directory Structure

```
scivianna_example/
├── demo.py                 # Main demonstrator entry point
├── demo_description.md     # Demonstrator description
├── llm_my_plot.py          # LLM integration through API example
├── image/                  # Demo images
├── c3po_coupling/          # C3PO coupling platform integration
├── europe_grid/            # Europe electricity visualization
├── mandelbrot/             # Mandelbrot fractal computation
└── med/                    # Salome MED file visualization
```

## Examples

### Europe Grid Example

Visualizes weekly electricity production and consumption data on a European map.

```python
from scivianna_example.europe_grid.europe_grid import make_europe_panel

panel = make_europe_panel(None)
panel.show()
```

**Features:**
- Interactive hover tools showing country-specific values
- Time-series navigation with weekly time-steps
- Production vs consumption comparison

See [`europe_grid/readme.md`](europe_grid/readme.md) for details.

### MED Coupling Example

Displays Salome MED file data with synchronized multi-view displays.

```python
from scivianna_example.med.split_item_example import get_panel

panel = get_panel("path/to/file.med")
panel.show()
```

**Features:**
- Three synchronized views of the same mesh
- Click interaction to offset views
- Split layout management

See [`med/readme.md`](med/readme.md) for details.

### Mandelbrot Example

Computes and visualizes the Mandelbrot fractal set.

```python
from scivianna_example.mandelbrot.mandelbrot import make_panel

panel = make_panel(None)
panel.show()
```

**Features:**
- Configurable resolution and iteration count
- Grid-to-polygon conversion
- Interactive zoom and pan controls

See [`mandelbrot/readme.md`](mandelbrot/readme.md) for details.

### C3PO Coupling Example

Integrates Scivianna with the C3PO coupling platform.

```python
from scivianna_example.c3po_coupling.fake_driver import run_fake_coupling

run_fake_coupling()
```

**Features:**
- ICOCO interface setup
- Real-time simulation visualization
- Field exchange using MED format

See [`c3po_coupling/readme.md`](c3po_coupling/readme.md) for details.

## Additional Resources

- Main Scivianna documentation: See the root `readme.md`
- API Reference: Check individual module docstrings
- Serialization examples: See `med/split_serialization.py` and `med/gridstack_serialization.py`
