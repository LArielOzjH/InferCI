"""Make the `inferci` package importable when running unittest discovery."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
