#!/bin/bash
###########################################################################
###   RUN_ALL_TESTS.SH                                                  ###
###      * Runs all test suites                                         ###
###      * Provides summary of test results                             ###
###########################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$BASE_DIR"

echo "================================================================================"
echo "Running All Tests"
echo "================================================================================"
echo ""

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

# Test 1: Python imports and syntax
echo "--------------------------------------------------------------------------------"
echo "Test 1: Python Imports and Syntax"
echo "--------------------------------------------------------------------------------"
if python3 "$SCRIPT_DIR/test_python_imports.py"; then
    echo "✓ Python tests passed"
    ((TESTS_PASSED++))
else
    echo "✗ Python tests failed"
    ((TESTS_FAILED++))
fi
echo ""

# Test 2: R syntax
echo "--------------------------------------------------------------------------------"
echo "Test 2: R Script Syntax"
echo "--------------------------------------------------------------------------------"
if Rscript "$SCRIPT_DIR/test_r_syntax.R"; then
    echo "✓ R syntax tests passed"
    ((TESTS_PASSED++))
else
    echo "✗ R syntax tests failed"
    ((TESTS_FAILED++))
fi
echo ""

# Test 3: File structure
echo "--------------------------------------------------------------------------------"
echo "Test 3: File Structure"
echo "--------------------------------------------------------------------------------"
if python3 "$SCRIPT_DIR/test_file_structure.py"; then
    echo "✓ File structure tests passed"
    ((TESTS_PASSED++))
else
    echo "✗ File structure tests failed"
    ((TESTS_FAILED++))
fi
echo ""

# Test 4: Compilation scripts (integration test)
echo "--------------------------------------------------------------------------------"
echo "Test 4: Compilation Script Integration"
echo "--------------------------------------------------------------------------------"
if Rscript "$SCRIPT_DIR/test_compilation_scripts.R"; then
    echo "✓ Compilation script tests passed"
    ((TESTS_PASSED++))
else
    echo "✗ Compilation script tests failed"
    ((TESTS_FAILED++))
fi
echo ""

# Summary
echo "================================================================================"
echo "Test Summary"
echo "================================================================================"
echo "Tests passed: $TESTS_PASSED"
echo "Tests failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo "✓ All tests passed!"
    exit 0
else
    echo "✗ Some tests failed"
    exit 1
fi
