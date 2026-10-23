# Graph Example

This example demonstrates scivianna's graph plotting: a `DataGraph` of 2D plots
connected by directed links, displayed as an interactive ReactFlow canvas built
by the `scivianna-graph` module.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization file |
| `graph.py` | Synthetic node builders (`lattice`, `hex_ring`, `bar`) and `make_graph()` |
| `graph_example.py` | Entry point: `make_graph_panel()` + `panel serve` support |
| `description.md` | Example description for the demonstrator |
| `readme.md` | This documentation file |

## Features shown

- **Drag nodes** individually around the canvas.
- **Rotate nodes**: the hex ring node is pre-rotated by 15 degrees (figure included).
- **Drag-select**: drag an empty area of the canvas to select several nodes at once, then move them together.
- **4-side plugs**: each link declares the side it attaches to on both endpoints
  (`source_direction` / `target_direction` in top/bottom/left/right), so nodes
  can be plugged on any side - not only left/right.
- **Hover** a node figure to see the cell value, as in the 2D panels.

## Run it

```bash
panel serve scivianna/src/scivianna_example/graph/graph_example.py --show
```

or from Python:

```python
from scivianna_example.graph.graph_example import make_graph_panel

make_graph_panel().main_frame.show()
```
