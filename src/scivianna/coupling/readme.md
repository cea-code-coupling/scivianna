# Scivianna Coupling Module

The `coupling` module provides integration with the C3PO coupling platform and implements the ICOCO interface for code-to-code coupling in simulation workflows.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization file |
| `icoco.py` | ICOCO-compatible interface implementation for coupling with C3PO |
| `problem_server.py` | Server-side problem definition for coupled simulations |
| `visualizer.py` | Visualizer-specific coupling utilities and helpers |

## Purpose

This module enables Scivianna to participate in multi-physics coupling scenarios where multiple simulation codes exchange data during runtime. It implements the ICOCO standard interface used by the C3PO coupling tool.

## Key Classes

### LayoutProblem
Main class implementing the ICOCO `Problem` interface:
- Manages visualization panels during coupling
- Handles time stepping synchronization
- Transfers field data between codes
- Supports stationary and transient modes

### ICoCo Interface Methods
| Method | Purpose |
|--------|---------|
| `initialize()` | Initialize the visualizer server |
| `terminate()` | Clean up resources and save state |
| `presentTime()` | Return current simulation time |
| `computeTimeStep()` | Suggest next time step |
| `initTimeStep(dt)` | Set time step for computation |
| `solveTimeStep()` | Perform visualization update |
| `validateTimeStep()` | Confirm time step completion |
| `setInputMEDDoubleField()` | Receive field data from coupling |

## Usage Example

An example of working coupling is available in the scivianna_example module, c3po_coupling

## Field Naming Convention

Fields are identified using the format: `panel_name@field_name`

Example: `"main_view@temperature"`

## Update Policies

Supported update policies for coupled fields:
- `UPDATE_DATA`: Update and replaces field values on existing mesh
- `UPDATE_MESH`: Update and replaces both mesh and field values
- `APPEND_DATA`: Add new time-step data
- `APPEND_MESH`: Add new time-step with new mesh

## Dependencies

- ICOCO library (~=2.0.0)
- MEDCoupling (for MED field handling)
- C3PO coupling platform (runtime)

## References

- ICOCO API: https://github.com/cea-trust-platform/icoco-coupling
- C3PO: https://github.com/code-coupling/c3po