# Contributing

Thanks for your interest in this project. This is primarily a learning project — a neural network library built from scratch using only NumPy — but contributions are welcome.

## What's useful

- Bug fixes (incorrect gradients, shape mismatches, numerical instability)
- New layer types (e.g. LSTM, embedding, attention)
- New examples that demonstrate existing functionality
- Documentation improvements

## What to avoid

- Adding framework dependencies (PyTorch, TensorFlow, JAX) — the point is pure NumPy
- Rewriting working code for style preferences
- Large refactors without discussion first

## How to contribute

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Test that existing examples still run (`python examples/xor_example.py`, `python examples/cnn_mnist_demo.py`)
4. Open a pull request with a clear description of what you changed and why

## Gradient correctness

If you add a new layer, verify its backward pass with the numerical gradient checker:

```bash
python examples/gradient_check.py
```

Relative error should stay below `1e-4`.

## Questions

Open an issue — happy to discuss ideas before you spend time building them.
