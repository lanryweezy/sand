"""
Pytest configuration and fixtures
"""

import sys
import os

# Add the silicon-intelligence directory to the path
# The directory name has a hyphen, so we need to import it differently
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the module using importlib
import importlib.util
spec = importlib.util.spec_from_file_location(
    "silicon_intelligence",
    os.path.join(os.path.dirname(__file__), '..', 'silicon-intelligence', '__init__.py')
)
silicon_intelligence = importlib.util.module_from_spec(spec)
sys.modules['silicon_intelligence'] = silicon_intelligence
spec.loader.exec_module(silicon_intelligence)

# Now import submodules
spec_data = importlib.util.spec_from_file_location(
    "silicon_intelligence.data",
    os.path.join(os.path.dirname(__file__), '..', 'silicon-intelligence', 'data', '__init__.py')
)
data_module = importlib.util.module_from_spec(spec_data)
sys.modules['silicon_intelligence.data'] = data_module
spec_data.loader.exec_module(data_module)

# Import the parser
spec_parser = importlib.util.spec_from_file_location(
    "silicon_intelligence.data.rtl_parser",
    os.path.join(os.path.dirname(__file__), '..', 'silicon-intelligence', 'data', 'rtl_parser.py')
)
parser_module = importlib.util.module_from_spec(spec_parser)
sys.modules['silicon_intelligence.data.rtl_parser'] = parser_module
spec_parser.loader.exec_module(parser_module)
