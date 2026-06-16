# Mandelbrot Example

This example demonstrates how to compute and visualize the Mandelbrot fractal set using Scivianna's polygon-based visualization capabilities.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization file |
| `mandelbrot.py` | Main module with Mandelbrot computation and panel creation |
| `mandelbrot.svg` | SVG icon of the Mandelbrot app extension |
| `description.md` | Example description for the demonstrator |
| `readme.md` | This documentation file |

## Overview

The Mandelbrot example showcases:
- Computing the Mandelbrot fractal set on a 2D grid
- Converting scalar field data to polygon representations
- Interactive visualization of fractal geometry
- Color mapping based on iteration count

## The Mandelbrot Set

The Mandelbrot set is defined as the set of complex numbers `c` for which the sequence:
```
z(n+1) = z(n)² + c, starting with z(0) = 0
```
remains bounded (does not diverge to infinity).

## Features

### Fractal Computation
- Configurable resolution and iteration count
- Smooth coloring based on iteration values

### Polygon Conversion
- Grid cells converted to polygons
- Contour-based visualization

### Interactive Display
- Zoom and pan controls
- Color map selection
- Iteration count adjustment

## Usage

run the `mandelbrot.py` file where the function make_panel builds an interactive layout.
