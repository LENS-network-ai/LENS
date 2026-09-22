"""
Ensures the repo root is importable as `model.*` / `training.*` / `utils.*`
when running `pytest` from anywhere in this project, without requiring
PYTHONPATH to be set manually first (unlike the training scripts, which do
rely on PYTHONPATH being set -- see the srun examples in README.md).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
