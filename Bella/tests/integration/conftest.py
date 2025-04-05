"""Test configuration for integration tests."""

import os
import sys
from pathlib import Path

# Add the project root to the Python path to make imports work
project_root = Path(__file__).parents[3]  # Go up 3 levels from this file
sys.path.insert(0, str(project_root))
