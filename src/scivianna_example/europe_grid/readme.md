# Europe Grid Example

This example demonstrates how to visualize geographic data using Scivianna, showing electricity production and consumption across European countries.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization file |
| `europe_grid.py` | Main module with Europe panel creation |
| `plot_api.py` | Plotting API for the Europe visualization |
| `country_time_series.py` | Time series data handling per country |
| `europe.geojson` | GeoJSON file with European country boundaries |
| `time_series.csv` | Weekly electricity production/consumption data |
| `description.md` | Example description for the demonstrator |
| `readme.md` | This documentation file |

## Overview

The Europe Grid example showcases:
- Geographic data visualization using GeoJSON
- Interactive hover tools showing country-specific values
- Time-series navigation with weekly time-steps
- Production vs consumption comparison
- Synchronized DataFrame panel showing aggregated country data

## Features

### Interactive Map
- Hover over countries to see electricity values
- Click to select specific countries
- Color-coded visualization of production/consumption

### Time Series Data
Weekly data including:
- Electricity production per energy source
- Electricity consumption per country
- Net imports/exports

### DataFrame Panel
- Automatically updates when hovering over countries
- Shows aggregated values for the selected country across all fields
- Linked to the 2D map and 1D time-series panels via `MOUSE_CELL_CHANGE` events

## Usage

### Open visualizer

Run the `europe_grid.py` file where the make_europe_panel builds an interactive layout.

## Plot API

The `plot_api.py` provides:
- Country selection callbacks
- Value formatting for tooltips
- Color scale configuration
