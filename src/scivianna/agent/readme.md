# Scivianna Agent Module

The `agent` module provides AI/LLM integration capabilities for Scivianna, enabling intelligent data manipulation and visualization assistance.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization file |
| `data_2d_worker.py` | Worker module for handling 2D data operations in agent context |
| `instructions.md` | Prompt instructions and examples for LLM agent interaction |
| `llm_model.py` | LLM model integration for AI-assisted visualization |

## Purpose

This module enables users to interact with Scivianna using natural language. The LLM agent can:

- Modify cell colors based on user requests
- Adjust transparency/alpha values to highlight or dim areas
- Filter and manipulate data based on value conditions
- Generate Python code to transform visualization properties

## Usage Example

```python
from scivianna.agent.data_2d_worker import LLMModel
from scivianna.data.data2d import Data2D

data_2d: Data2D = ...
# Create agent with access to 2D data
agent = Data2DWorker(data_2d)

# Request natural language transformations
agent("Color in red the highest value cell")
agent("Hide zeros, highlight highest values, dim the rest")
```

## Key Features

- **Data2D Access**: Agent has programmatic access to cell values, colors, and alpha channels
- **Code Generation**: Generates executable Python code snippets for data manipulation
- **NumPy Integration**: Full NumPy support for array operations
- **Instruction Templates**: Pre-defined instruction patterns for common tasks
