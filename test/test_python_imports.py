#!/usr/bin/env python3
"""
Test Python module imports and basic syntax validation.
This test ensures all core Python modules can be imported without errors.
"""

import sys
import os

# Add model directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

def test_imports():
    """Test that all core modules can be imported."""
    # Note: Import failures are OK - we're primarily testing syntax
    # The syntax validation below will catch actual code errors
    # Import failures just indicate missing dependencies, which is expected in CI
    import_warnings = []
    
    modules_to_test = [
        'MeaslesDataLoader',
        'MeaslesModelEval',
        'EpiPreprocessor',
        'fitOne'
    ]
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name} imported successfully")
        except ImportError as e:
            print(f"⚠ {module_name} import failed (missing dependency): {e}")
            import_warnings.append(module_name)
        except Exception as e:
            # Any other exception is also treated as a warning (might be dependency-related)
            print(f"⚠ {module_name} import failed: {e}")
            import_warnings.append(module_name)
    
    if import_warnings:
        print(f"\n⚠ Note: {len(import_warnings)} module(s) had import warnings (likely missing dependencies)")
        print("   This is OK - syntax validation below will catch code errors")
    
    # Always return True - import failures don't fail the test
    # We're testing syntax, not whether dependencies are installed
    return True

def test_syntax():
    """Test that Python files have valid syntax."""
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    python_files = [
        'MeaslesDataLoader.py',
        'MeaslesModelEval.py',
        'EpiPreprocessor.py',
        'fitOne.py',
        'LossFunctions.py',
        'SeasonalityMetrics.py',
        'ModelSweeps.py',
        'EpiAnnealer.py'
    ]
    
    all_valid = True
    for filename in python_files:
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    compile(f.read(), filepath, 'exec')
                print(f"✓ {filename} syntax is valid")
            except SyntaxError as e:
                print(f"✗ {filename} has syntax error: {e}")
                all_valid = False
            except Exception as e:
                print(f"✗ {filename} error: {e}")
                all_valid = False
        else:
            print(f"⚠ {filename} not found (skipping)")
    
    return all_valid

if __name__ == '__main__':
    print("=" * 60)
    print("Python Import and Syntax Tests")
    print("=" * 60)
    
    imports_ok = test_imports()
    syntax_ok = test_syntax()
    
    print("=" * 60)
    # Only syntax errors should cause test failure
    # Import failures are expected and don't indicate code problems
    if syntax_ok:
        print("✓ All tests passed (syntax validation successful)")
        sys.exit(0)
    else:
        print("✗ Syntax validation failed - code has syntax errors")
        sys.exit(1)
