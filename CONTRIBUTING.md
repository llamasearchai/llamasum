# Contributing to LlamaSum

Thank you for considering contributing to LlamaSum! This document outlines the process for contributing to the project and provides guidelines to ensure a smooth collaboration.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## How Can I Contribute?

### Reporting Bugs

When reporting bugs, please include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, package versions)
- Any relevant logs or error messages

Please use the GitHub issue tracker with the "bug" template.

### Suggesting Enhancements

When suggesting enhancements, please include:

- A clear and descriptive title
- A detailed description of the proposed functionality
- Any potential implementation details you have in mind
- Why this enhancement would be useful to most users

Please use the GitHub issue tracker with the "feature request" template.

### Pull Requests

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `pip install -e ".[dev]"`
4. Make your changes
5. Run tests: `pytest`
6. Run linting: `pre-commit run --all-files`
7. Commit your changes: `git commit -m 'Add some amazing feature'`
8. Push to the branch: `git push origin feature/amazing-feature`
9. Open a Pull Request

## Development Setup

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/llamasum.git
   cd llamasum
   ```

2. Set up a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks
   ```bash
   pre-commit install
   ```

## Testing

We use pytest for testing. To run the tests:

```bash
pytest
```

To run with coverage:

```bash
pytest --cov=llamasum
```

## Code Style

We use:
- Black for formatting
- Flake8 for linting
- isort for sorting imports
- mypy for type checking

These checks are enforced through pre-commit hooks.

## Documentation

We use Google style docstrings for documentation:

```python
def function_name(param1, param2):
    """Short description of function.

    More detailed description if needed.

    Args:
        param1 (type): Description of param1.
        param2 (type): Description of param2.

    Returns:
        return_type: Description of return value.

    Raises:
        ExceptionType: When and why this exception is raised.
    """
    # Function implementation
```

## Git Commit Guidelines

We follow the Conventional Commits specification:

- `feat:` - A new feature
- `fix:` - A bug fix
- `docs:` - Documentation only changes
- `style:` - Changes that don't affect code meaning (formatting, etc.)
- `refactor:` - Code change that neither fixes a bug nor adds a feature
- `perf:` - Code change that improves performance
- `test:` - Adding or correcting tests
- `chore:` - Changes to build process or auxiliary tools

Example: `feat: add hierarchical summarization capability`

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Commit these changes: `git commit -m "chore: prepare release X.Y.Z"`
4. Create a tag: `git tag vX.Y.Z`
5. Push changes and tag: `git push && git push --tags`
6. Build and publish to PyPI:
   ```bash
   python -m build
   python -m twine upload dist/*
   ```

## Questions?

If you have any questions, feel free to open an issue with the "question" label or contact the maintainers directly.

Thank you for contributing to LlamaSum! 