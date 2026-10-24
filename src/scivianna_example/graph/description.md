##   Graph example

In this example, a hierarchical 1D pipe network (inlet, three parallel branches - one of them with elbows, and a nested pair of parallel pipes - then a collector and an outlet) is laid out and displayed as an interactive ReactFlow canvas.

Each pipe is a Bokeh figure fitted to its content; the flow enters every pipe through its left plug and leaves through its right plug in the node's own frame, so elbows (rotated 90 degrees) chain correctly. Nodes can be dragged individually or selected in groups by dragging an empty area of the canvas, and each link is plugged on a specific side. Hovering a pipe shows the cell value, as in the 2D panels.
