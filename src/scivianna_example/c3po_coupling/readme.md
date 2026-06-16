# C3PO Coupling Example

This example demonstrates how to use Scivianna with the C3PO coupling platform for real-time simulation visualization.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization file |
| `coupling.py` | Main coupling script with ICOCO interface setup |
| `fake_driver.py` | Mock driver for testing without actual simulation codes |
| `readme.md` | This documentation file |

## Overview

The C3PO coupling example shows how to:
- Set up a Scivianna visualizer as an ICOCO problem
- Connect to C3PO for multi-physics coupling
- Receive and display field updates during simulation
- Synchronize visualization with simulation time steps

## Architecture

```
┌─────────────┐      ┌──────────────┐     ┌─────────────┐
│   C3PO      │────▶│  ICOCO       │────▶│  Scivianna  │
│   Coupler   │      │  Interface   │     │  Visualizer │
└─────────────┘      └──────────────┘     └─────────────┘
```

## Field Exchange

Fields are exchanged using MED format with naming convention:
- `"panel_name@field_name"`

Example:
```python
problem.setInputMEDDoubleField("main_view@temperature", med_field)
```

## Requirements

- C3PO coupling platform
- ICOCO library
- Salome MEDCoupling