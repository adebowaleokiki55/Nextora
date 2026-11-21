"""Nextora backend package.

Thread limits are set here, before any submodule imports numpy, torch or
OpenCV, so the small Render instance does not oversubscribe its CPU.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
