# Contributing to Berlin Grid Project

Thank you for your interest in contributing to the Berlin Grid Real-Time Simulation Framework! This document provides guidelines for contributing to this real-time grid digital twin project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful, collaborative, and professional environment for all contributors.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When reporting a bug, include:

- **Clear and descriptive title**
- **Detailed steps to reproduce** the issue
- **Expected behavior vs. actual behavior**
- **Environment details**: Python version, OS, hardware specs
- **System performance metrics** if relevant (loop times, memory usage)
- **Log files or error messages**
- **Sample data** if applicable (anonymize sensitive information)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When suggesting an enhancement, include:

- **Clear and descriptive title**
- **Detailed description** of the proposed enhancement
- **Use cases and benefits**
- **Technical feasibility considerations**
- **Impact on real-time performance** (<50ms physics loop constraint)
- **Examples from other projects** if relevant

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the code style guidelines
3. **Add tests** for any new functionality
4. **Ensure all tests pass** including unit tests and integration tests
5. **Update documentation** if you're changing functionality
6. **Verify real-time performance** constraints are maintained
7. **Write clear commit messages** describing your changes
8. **Submit a pull request** with a comprehensive description

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Virtual environment tool (venv or conda)
- Git
- Understanding of power systems engineering (beneficial)

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/berlin-grid-project.git
cd berlin-grid-project

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test suite
python -m pytest tests/test_grid_simulation.py -v

# Run performance tests
python -m pytest tests/test_performance.py -v
```

## Code Style Guidelines

### General Guidelines

- Follow **PEP 8** style guidelines
- Use **type hints** for all function parameters and return values
- Write **docstrings** for all classes and functions (Google style)
- Keep functions **focused and modular**
- Use **meaningful variable and function names**
- Add **inline comments** for complex logic

### Naming Conventions

```python
# Classes: PascalCase
class GridSimulator:
    pass

# Functions and variables: snake_case
def calculate_power_flow(voltage: float, current: float) -> float:
    power_value = voltage * current
    return power_value

# Constants: UPPER_SNAKE_CASE
MAX_VOLTAGE_KV = 110
PHYSICS_LOOP_MS = 50
```

### Documentation

```python
def validate_vde_compliance(
    grid_state: GridState,
    voltage_limits: VoltageLimit
) -> ComplianceResult:
    """
    Validate grid state against VDE-AR-N 4110 compliance requirements.
    
    Args:
        grid_state: Current state of the grid system
        voltage_limits: Acceptable voltage range per VDE standards
        
    Returns:
        ComplianceResult containing validation status and violations
        
    Raises:
        ValidationError: If grid state data is invalid
    """
    pass
```

## Testing Guidelines

### Unit Tests

- Write unit tests for **all new functionality**
- Aim for **>80% test coverage**
- Use **descriptive test names** that explain what is being tested
- Include both **positive and negative test cases**
- Mock external dependencies appropriately

### Performance Tests

- Verify **<50ms physics loop** constraint is maintained
- Test with **realistic grid sizes** (substations, nodes)
- Profile **memory usage** for long-running simulations
- Test **Monte Carlo simulation** performance (n=50 iterations)

### Integration Tests

- Test **IEC 61850 protocol** integration
- Verify **VDE-AR-N 4110 compliance** validation
- Test **SCADA integration** workflows
- Validate **data persistence** and retrieval

## Performance Requirements

This project has strict real-time performance requirements:

- **Physics loop**: Must complete in <50ms
- **Memory efficiency**: Minimize allocations in hot paths
- **Profiling**: Use `cProfile` or `py-spy` for optimization
- **Benchmarking**: Include performance benchmarks for critical paths

```bash
# Profile your changes
python -m cProfile -o profile.stats main.py

# Analyze with snakeviz
pip install snakeviz
snakeviz profile.stats
```

## Domain-Specific Considerations

### Power Systems Knowledge

- Understand **VDE-AR-N 4110** hosting capacity requirements
- Familiarize yourself with **IEC 61850** protocol standards
- Review **redispatch** and grid congestion management concepts
- Understand **Monte Carlo simulation** for uncertainty modeling

### Electrical Engineering Conventions

- Use **per-unit (p.u.)** system for calculations when appropriate
- Follow **IEEE standards** for power system notation
- Include **unit tests** for physical calculations (Ohm's law, power flow)
- Document **assumptions** about grid topology and parameters

## Commit Message Guidelines

Write clear and meaningful commit messages:

```
# Format
<type>: <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat: Add IEC 61850 substation integration layer

- Implement MMS protocol client
- Add data mapping for logical nodes
- Include integration tests for protocol compliance

Closes #45
```

```
perf: Optimize physics loop to achieve <40ms cycle time

- Vectorize power flow calculations using NumPy
- Cache frequently accessed grid parameters
- Reduce memory allocations in hot path

Benchmark: 48ms -> 38ms (21% improvement)
```

## Pull Request Process

1. **Update documentation** for any changed functionality
2. **Update CHANGELOG.md** with your changes
3. **Ensure CI/CD pipeline passes** all checks
4. **Request review** from at least one maintainer
5. **Address review feedback** promptly
6. **Squash commits** if requested before merging

## Documentation

When contributing documentation:

- Update **README.md** for user-facing changes
- Update **API documentation** for code changes
- Add **examples** for new features
- Include **diagrams** for complex architectures (use Mermaid)
- Document **performance characteristics** of new features

## Questions and Support

If you have questions about contributing:

- **Open an issue** for general questions
- **Start a discussion** for broader topics
- **Check existing documentation** in `/docs` directory
- **Review closed issues** for similar questions

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Thank you for taking the time to contribute! Your efforts help advance real-time grid simulation for Germany's energy transition.

## Additional Resources

- [VDE-AR-N 4110 Standard](https://www.vde.com/de/fnn/arbeitsgebiete/tar/tar-niederspannung/vde-ar-n-4110)
- [IEC 61850 Documentation](https://en.wikipedia.org/wiki/IEC_61850)
- [PandaPower Documentation](https://pandapower.readthedocs.io/)
- [Real-Time Systems Best Practices](https://realtimesystems.org/)
