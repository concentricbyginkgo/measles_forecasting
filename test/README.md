# Test Suite

This directory contains integration tests for the measles forecasting repository.

## Test Structure

- **`test_python_imports.py`** - Tests Python module imports and syntax validation
- **`test_r_syntax.R`** - Tests R script syntax validation
- **`test_file_structure.py`** - Validates required directories and files exist
- **`test_compilation_scripts.R`** - Integration tests for model output compilation scripts
- **`run_all_tests.sh`** - Master script to run all tests

## Running Tests

### Run All Tests
```bash
./test/run_all_tests.sh
```

### Run Individual Tests

**Python tests:**
```bash
python test/test_python_imports.py
python test/test_file_structure.py
```

**R tests:**
```bash
Rscript test/test_r_syntax.R
Rscript test/test_compilation_scripts.R
```

## Test Data

The `test/data/` directory contains sample data files used for integration testing:
- Sample metadata files (column **`geography`**, not `country`; matches [`fitOne.py`](../model/fitOne.py))
- Sample summary statistics
- Sample time series projections

## Continuous Integration

Tests are automatically run on GitHub Actions when:
- Code is pushed to `main` or `develop` branches
- Pull requests are opened

See `.github/workflows/build-test.yml` for CI configuration.

## Adding New Tests

When adding new functionality:

1. **Python modules**: Add import tests to `test_python_imports.py`
2. **R scripts**: Add syntax tests to `test_r_syntax.R`
3. **New workflows**: Add integration tests to `test_compilation_scripts.R` or create new test files
4. **File structure**: Update `test_file_structure.py` if new required files/directories are added

## Test Requirements

- Python 3.11+
- R 4.0+
- Required packages (see main README.md for full list)
