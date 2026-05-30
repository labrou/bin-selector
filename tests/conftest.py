import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import (
    compute_plurality, compute_abs_majority, compute_weighted,
    compute_view, dim_color, make_view_csv, generate_data,
    VARIOUS_LABEL, ITEMS, COLORS, OTHER_COLOR, BG, METHOD_OPTIONS
)
