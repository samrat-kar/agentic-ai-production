# Contributing to RAG Research Assistant

Thank you for considering contributing to this project! Here's how you can help.

## How to Contribute

1. **Fork** the repository and create your branch from `main`.
2. **Install** dev dependencies: `pip install -r requirements-dev.txt`
3. **Make** your changes with clear, descriptive commit messages.
4. **Add tests** for any new functionality in the `tests/` directory.
5. **Run the test suite** to make sure nothing is broken:
   ```bash
   pytest tests/ -v
   ```
6. **Lint** your code:
   ```bash
   ruff check src/ tests/
   ```
7. **Open a Pull Request** against `main` with a clear description of your changes.

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use type hints for function signatures.
- Add docstrings to all public functions and classes.
- This project uses **Ruff** for linting — see `pyproject.toml` for configuration.

## Reporting Issues

- Use GitHub Issues to report bugs or request features.
- Include reproduction steps, expected behavior, and actual behavior.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).
