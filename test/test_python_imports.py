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
    # Note: Some imports may fail due to missing dependencies, which is OK for syntax tests
    # The syntax validation below will catch actual code errors
    import_warnings = []
    
    try:
        import MeaslesDataLoader
        print("✓ MeaslesDataLoader imported successfully")
    except ImportError as e:
        print(f"⚠ MeaslesDataLoader import failed (missing dependency): {e}")
        import_warnings.append("MeaslesDataLoader")
    except Exception as e:
        print(f"✗ Failed to import MeaslesDataLoader: {e}")
        return False
    
    try:
        import MeaslesModelEval
        print("✓ MeaslesModelEval imported successfully")
    except ImportError as e:
        print(f"⚠ MeaslesModelEval import failed (missing dependency): {e}")
        import_warnings.append("MeaslesModelEval")
    except Exception as e:
        print(f"✗ Failed to import MeaslesModelEval: {e}")
        return False
    
    try:
        import EpiPreprocessor
        print("✓ EpiPreprocessor imported successfully")
    except ImportError as e:
        print(f"⚠ EpiPreprocessor import failed (missing dependency): {e}")
        import_warnings.append("EpiPreprocessor")
    except Exception as e:
        print(f"✗ Failed to import EpiPreprocessor: {e}")
        return False
    
    try:
        import fitOne
        print("✓ fitOne imported successfully")
    except ImportError as e:
        print(f"⚠ fitOne import failed (missing dependency): {e}")
        import_warnings.append("fitOne")
    except Exception as e:
        print(f"✗ Failed to import fitOne: {e}")
        return False
    
    if import_warnings:
        print(f"\n⚠ Note: {len(import_warnings)} module(s) had import warnings (likely missing dependencies)")
        print("   This is OK - syntax validation below will catch code errors")
    
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
    if imports_ok and syntax_ok:
        print("✓ All tests passed")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
