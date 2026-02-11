# Contributing to scModelForge

Thank you for your interest in contributing to scModelForge! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork and clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/scModelForge.git
   cd scModelForge
   ```

2. Create a virtual environment and install in development mode:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.
- Line length limit is 120 characters.
- Python 3.10+ features are encouraged (union types with `|`, `match` statements, etc.).
- Run checks locally before pushing:

  ```bash
  ruff check src/ tests/
  ruff format src/ tests/
  ```

## Testing

- Write tests for all new functionality.
- Use `pytest` to run the test suite:

  ```bash
  pytest                              # Run all tests (excluding slow/gpu)
  pytest -m "not slow and not gpu"    # Same as above (explicit)
  pytest tests/test_smoke.py -v       # Run a specific test file
  ```

- Mark slow tests with `@pytest.mark.slow` and GPU tests with `@pytest.mark.gpu`.
- Shared fixtures are in `tests/conftest.py`.

## Pull Request Process

1. Create a feature branch from `main`.
2. Make your changes with clear, descriptive commits.
3. Ensure all tests pass and linting is clean.
4. Submit a pull request with a clear description of the changes.

## Project Structure

```
src/scmodelforge/
├── data/          # Data loading and preprocessing
├── tokenizers/    # Tokenization strategies
├── models/        # Model architectures
├── training/      # Training loop and utilities
├── eval/          # Evaluation benchmarks
├── config/        # Configuration schema
└── cli.py         # Command-line interface
```

## License

By contributing to scModelForge, you agree that your contributions will be licensed under the Apache 2.0 License.
