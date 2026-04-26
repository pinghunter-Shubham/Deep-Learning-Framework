# Deep Learning Framework

A mini deep learning framework built from scratch, implementing core machine learning concepts including automatic differentiation, neural networks, and optimization algorithms.

## Project Overview

This project provides a lightweight deep learning framework with the following features:
- **Autograd System**: Automatic differentiation for computing gradients
- **Neural Network Modules**: Layers and network architectures
- **Optimization**: SGD and Adam optimizers
- **Operations**: Tensor operations and computations
- **MNIST Training**: Example training script for MNIST dataset

## Project Structure

```
DEEPLEARNING FRAMEWORK/
├── .git/                          # Git repository metadata
├── .gitignore                     # Git ignore rules
├── .pytest_cache/                 # Pytest cache files
├── .vscode/                       # VS Code configuration
├── data/                          # MNIST dataset files
│   ├── t10k-images-idx3-ubyte.gz  # Test images
│   ├── t10k-labels-idx1-ubyte.gz  # Test labels
│   ├── train-images-idx3-ubyte.gz # Training images
│   └── train-labels-idx1-ubyte.gz # Training labels
├── minitorch/                     # Main package
│   ├── __init__.py                # Package initialization
│   ├── autograd.py                # Automatic differentiation system
│   ├── dataset.py                 # Dataset utilities
│   ├── ops.py                     # Tensor operations
│   ├── storage.py                 # Storage backend
│   ├── tensor.py                  # Tensor core class
│   ├── testing.py                 # Testing utilities
│   ├── nn/                        # Neural network modules
│   │   ├── __init__.py
│   │   ├── linear.py              # Linear/Dense layer
│   │   ├── loss.py                # Loss functions
│   │   ├── module.py              # Base module class
│   │   └── sequential.py          # Sequential container
│   └── optim/                     # Optimization algorithms
│       ├── __init__.py
│       ├── adam.py                # Adam optimizer
│       ├── optimizer.py           # Base optimizer class
│       └── sgd.py                 # SGD optimizer
├── tests/                         # Test suite
│   ├── test_gradcheck.py          # Gradient checking tests
│   ├── test_nn.py                 # Neural network tests
│   └── test_ops.py                # Operations tests
├── debug.py                       # Debug script 1
├── debug2.py                      # Debug script 2
├── debug3.py                      # Debug script 3
├── debug4.py                      # Debug script 4
├── train_mnist.py                 # MNIST training script
├── test_out.txt                   # Test output
└── pyproject.toml                 # Project configuration

```

## Module Descriptions

### Core Modules (`minitorch/`)

| File | Purpose |
|------|---------|
| `tensor.py` | Core Tensor class with gradient computation support |
| `autograd.py` | Automatic differentiation engine (computational graph tracking) |
| `ops.py` | Tensor operations and mathematical functions |
| `storage.py` | Backend storage for tensor data |
| `dataset.py` | Data loading and preprocessing utilities |
| `testing.py` | Testing and debugging utilities |

### Neural Network (`minitorch/nn/`)

| File | Purpose |
|------|---------|
| `module.py` | Base `Module` class for all network components |
| `linear.py` | Fully connected (Dense) layer implementation |
| `sequential.py` | Sequential container for stacking layers |
| `loss.py` | Loss functions (e.g., Cross-Entropy, MSE) |

### Optimization (`minitorch/optim/`)

| File | Purpose |
|------|---------|
| `optimizer.py` | Base `Optimizer` class |
| `sgd.py` | Stochastic Gradient Descent optimizer |
| `adam.py` | Adam optimizer implementation |

## Main Scripts

| File | Purpose |
|------|---------|
| `train_mnist.py` | Main training script for MNIST digit classification |
| `debug.py` | Debugging/testing script 1 |
| `debug2.py` | Debugging/testing script 2 |
| `debug3.py` | Debugging/testing script 3 |
| `debug4.py` | Debugging/testing script 4 |

## Dependencies

The project requires the following Python packages (see `pyproject.toml`):
- **numpy**: Numerical computations
- **pytest**: Testing framework

Install dependencies with:
```bash
pip install -r requirements.txt
```

Or using pip:
```bash
pip install numpy pytest
```

## Getting Started

### Training the Model

Run the MNIST training script:
```bash
python train_mnist.py
```

### Running Tests

Execute the test suite:
```bash
pytest tests/
```

Or run specific tests:
```bash
pytest tests/test_nn.py
pytest tests/test_ops.py
pytest tests/test_gradcheck.py
```

## Dataset

The project includes the MNIST dataset in the `data/` directory:
- **Training**: 60,000 images (28×28 pixels)
- **Testing**: 10,000 images (28×28 pixels)
- **Classes**: 10 digits (0-9)

## Key Features

✅ **Automatic Differentiation**: Compute gradients automatically  
✅ **Neural Network Layers**: Implement custom architectures  
✅ **Multiple Optimizers**: SGD and Adam  
✅ **Loss Functions**: Cross-entropy, MSE, etc.  
✅ **MNIST Example**: Pre-built training pipeline  
✅ **Comprehensive Tests**: Full test coverage  

## Repository

This project is hosted on GitHub: [Deep-Learning-Framework](https://github.com/pinghunter-Shubham/Deep-Learning-Framework)

## License

This is an educational project for learning deep learning concepts from scratch.

---

**Last Updated**: April 26, 2026
