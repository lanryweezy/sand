import sys
import os
import unittest

# Add the project root to sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Discover and run tests
# -s: start directory (relative to current directory)
# -p: pattern to match test files
# -t: top level directory (where to look for imports)
loader = unittest.TestLoader()
suite = loader.discover(start_dir='silicon_intelligence/tests', pattern='test_*.py', top_level_dir=project_root)

runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
