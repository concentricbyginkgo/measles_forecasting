#!/usr/bin/env python3
"""
Test file structure and directory organization.
Validates that expected directories and files exist.
"""

import os
import sys

def test_file_structure():
    """Test that required directories and files exist."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 60)
    print("File Structure Tests")
    print("=" * 60)
    
    all_passed = True
    
    # Required directories
    required_dirs = [
        'model',
        'model/input',
        'model_output_processing',
        'data_ingestion_pipeline',
        'grid_search',
        'shiny_standalone',
        'shiny_standalone/data',
    ]
    
    print("\nChecking required directories...")
    for dir_path in required_dirs:
        full_path = os.path.join(base_dir, dir_path)
        if os.path.isdir(full_path):
            print(f"✓ {dir_path}/ exists")
        else:
            print(f"✗ {dir_path}/ missing")
            all_passed = False
    
    # Required files
    required_files = [
        'model/MeaslesDataLoader.py',
        'model/MeaslesModelEval.py',
        'model/fitOne.py',
        'model_output_processing/1_compile_summary_table.R',
        'model_output_processing/2_compile_time_series_tables.R',
        'shiny_standalone/global.R',
        'shiny_standalone/server.R',
        'shiny_standalone/ui.R',
        'README.md',
    ]
    
    print("\nChecking required files...")
    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        if os.path.isfile(full_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            all_passed = False
    
    # Check for metadata file (should be in model/input/)
    metadata_path = os.path.join(base_dir, 'model/input/metadata_example.csv')
    if os.path.isfile(metadata_path):
        print(f"✓ model/input/metadata_example.csv exists")
    else:
        # Check old location
        old_path = os.path.join(base_dir, 'grid_search/metadata_example.csv')
        if os.path.isfile(old_path):
            print(f"⚠ metadata_example.csv found in old location (grid_search/)")
        else:
            print(f"⚠ metadata_example.csv not found (may need to be created)")
    
    print("=" * 60)
    if all_passed:
        print("✓ All file structure tests passed")
        return True
    else:
        print("✗ Some file structure tests failed")
        return False

if __name__ == '__main__':
    success = test_file_structure()
    sys.exit(0 if success else 1)
