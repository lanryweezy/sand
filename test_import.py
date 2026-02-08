#!/usr/bin/env python3
"""Test script to check imports"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

print("Testing imports...")

# First, let's check if the module can be imported at all
try:
    import models.drc_predictor as drc_module
    print("✓ Module imported successfully")
    print(f"Module file: {drc_module.__file__}")
    print(f"Module attributes: {[attr for attr in dir(drc_module) if not attr.startswith('_')]}")
except Exception as e:
    print(f"✗ Module import failed: {e}")
    import traceback
    traceback.print_exc()

# Now try direct import
try:
    from models.drc_predictor import DRCPredictor, DRCAwarePlacer
    print("✓ Direct import successful")
    print(f"DRCPredictor: {DRCPredictor}")
    print(f"DRCAwarePlacer: {DRCAwarePlacer}")
except Exception as e:
    print(f"✗ Direct import failed: {e}")
    import traceback
    traceback.print_exc()