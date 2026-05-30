"""Tests for generate_data."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core import (
    generate_data, ITEMS, NUM_BINS, NUM_DATES, NUM_POSITIONS,
)

N_ITEMS = len(ITEMS)


def test_generate_data_keys():
    """All expected keys are present in the returned dict."""
    data = generate_data()
    expected_keys = {
        'date_winner', 'date_top_share',
        'wt_bin_idx', 'wt_date_idx', 'wt_pos_idx', 'wt_item_idx', 'wt_N_item',
        'bin_ranks', 'bin_segments', 'bin_names',
        'dates', 'positions', 'item_codes', 'item_colors', '_id',
    }
    assert expected_keys.issubset(set(data.keys()))


def test_generate_data_shapes():
    """date_winner shape == (NUM_BINS, NUM_DATES, NUM_POSITIONS)."""
    data = generate_data()
    assert data['date_winner'].shape == (NUM_BINS, NUM_DATES, NUM_POSITIONS)
    assert data['date_top_share'].shape == (NUM_BINS, NUM_DATES, NUM_POSITIONS)


def test_generate_data_winner_range():
    """Winners are in [-1, len(ITEMS)-1]."""
    data = generate_data()
    w = data['date_winner']
    assert w.min() >= -1
    assert w.max() <= N_ITEMS - 1


def test_generate_data_share_range():
    """top_share values are in [0, 1]."""
    data = generate_data()
    s = data['date_top_share']
    assert float(s.min()) >= 0.0
    assert float(s.max()) <= 1.0 + 1e-5  # allow tiny fp rounding


def test_generate_data_reproducible():
    """Two calls give identical results (fixed seed in generate_data)."""
    d1 = generate_data()
    d2 = generate_data()
    np.testing.assert_array_equal(d1['date_winner'],    d2['date_winner'])
    np.testing.assert_array_equal(d1['date_top_share'], d2['date_top_share'])
    np.testing.assert_array_equal(d1['wt_N_item'],      d2['wt_N_item'])


def test_generate_data_sparse_arrays():
    """wt_bin_idx, wt_date_idx, wt_pos_idx, wt_item_idx, wt_N_item all have the same length."""
    data = generate_data()
    lengths = {
        'wt_bin_idx':  len(data['wt_bin_idx']),
        'wt_date_idx': len(data['wt_date_idx']),
        'wt_pos_idx':  len(data['wt_pos_idx']),
        'wt_item_idx': len(data['wt_item_idx']),
        'wt_N_item':   len(data['wt_N_item']),
    }
    values = list(lengths.values())
    assert all(v == values[0] for v in values), f"Mismatched lengths: {lengths}"
