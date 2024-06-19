# Linear Regression Model

This repository contains two main scripts: one for creating and training a linear regression model using PyTorch, and another for testing the model using pytest.

## Files

1. **`regression.py`** - Contains functions to create, train, and fit a linear regression model.
2. **`test_regression.py`** - Contains tests for the linear regression model using pytest.

## Prerequisites

- Python 3.x
- `torch` library
- `pytest` library

Install the required libraries:

```sh
pip install torch pytest
```

## Usage

### Training the Model

1. Ensure you have your input and output tensors `X` and `y` ready.
2. Import the `fit_regression_model` function from `regression.py` and call it with your data.

Example usage:

```python
from regression import fit_regression_model
import torch

# Example input and output tensors
X = torch.tensor([[24.], [16.], [25.], [19.], [14.]])
y = torch.tensor([[1422.4], [1012.7], [1632.2], [1117.7], [906.2]])

# Train the model
trained_model, final_loss = fit_regression_model(X, y)
```

### Running Tests

1. Ensure `regression.py` and `test_regression.py` are in your working directory.
2. Run the tests using pytest:

```sh
pytest test_regression.py
```

## File Overview

### `regression.py`

Defines functions to create, train, and fit a linear regression model using PyTorch.

### `test_regression.py`

Defines tests for the linear regression model using pytest, including:
- Training the model on 1D and 2D data.
- Checking the prediction accuracy.

## Example Tests

```sh
pytest test_regression.py
```

