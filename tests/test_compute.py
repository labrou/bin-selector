"""Tests for compute_plurality, compute_abs_majority, compute_weighted, compute_view."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core import (
    compute_plurality, compute_abs_majority, compute_weighted,
    compute_view, generate_data, ITEMS,
)

N_ITEMS = len(ITEMS)
VARIOUS_IDX = N_ITEMS  # sentinel beyond real item indices


# ── compute_plurality ──────────────────────────────────────────────────────────

def test_plurality_single_date():
    """Shape (2,1,3): single-date, winner == the single date's winner."""
    dw = np.array([[[0, 1, 2]], [[2, 1, 0]]], dtype=np.int32)
    winner, share = compute_plurality(dw, n_items=3)
    assert winner.shape == (2, 3)
    np.testing.assert_array_equal(winner[0], [0, 1, 2])
    np.testing.assert_array_equal(winner[1], [2, 1, 0])
    # single date → share is 1.0 wherever data exists
    np.testing.assert_array_equal(share[0], [1.0, 1.0, 1.0])


def test_plurality_clear_winner():
    """3 dates, same winner every time → share = 1.0."""
    # 1 bin, 3 dates, 2 positions; item 0 always wins
    dw = np.array([[[0, 1], [0, 1], [0, 1]]], dtype=np.int32)
    winner, share = compute_plurality(dw, n_items=2)
    np.testing.assert_array_equal(winner[0], [0, 1])
    np.testing.assert_array_almost_equal(share[0], [1.0, 1.0])


def test_plurality_tiebreak():
    """Two items tied on win count; most-recent-date winner should win."""
    # 1 bin, 2 dates, 1 position
    # date 0: item 0 wins; date 1: item 1 wins → tie (1 win each)
    # most recent date (index 1) had item 1 → item 1 should win tiebreak
    dw = np.array([[[0], [1]]], dtype=np.int32)
    winner, share = compute_plurality(dw, n_items=2)
    assert winner[0, 0] == 1


def test_plurality_no_data():
    """All -1 → winner = -1, share = 0."""
    dw = np.full((2, 3, 4), -1, dtype=np.int32)
    winner, share = compute_plurality(dw, n_items=3)
    assert np.all(winner == -1)
    assert np.all(share == 0.0)


def test_plurality_shape():
    """Output shape == (n_bins, n_pos)."""
    n_bins, n_dates, n_pos = 5, 7, 10
    dw = np.random.randint(0, 3, (n_bins, n_dates, n_pos), dtype=np.int32)
    winner, share = compute_plurality(dw, n_items=3)
    assert winner.shape == (n_bins, n_pos)
    assert share.shape == (n_bins, n_pos)


# ── compute_abs_majority ───────────────────────────────────────────────────────

def test_abs_majority_above_threshold():
    """Share >= 0.5: winner is the plurality winner (item 0)."""
    dw  = np.array([[[0, 1]]], dtype=np.int32)
    dts = np.array([[[0.6, 0.7]]], dtype=np.float32)
    winner, share = compute_abs_majority(dw, dts, n_items=2, various_idx=2)
    assert winner[0, 0] == 0
    assert winner[0, 1] == 1


def test_abs_majority_below_threshold():
    """Share < 0.5 → VARIOUS (various_idx)."""
    dw  = np.array([[[0]]], dtype=np.int32)
    dts = np.array([[[0.4]]], dtype=np.float32)
    winner, share = compute_abs_majority(dw, dts, n_items=2, various_idx=2)
    assert winner[0, 0] == 2  # VARIOUS


def test_abs_majority_various_can_win():
    """VARIOUS can win the cross-date majority vote."""
    # 3 dates, 1 bin, 1 position:
    # date 0: share=0.3 → VARIOUS (idx 2)
    # date 1: share=0.3 → VARIOUS (idx 2)
    # date 2: share=0.9 → item 1
    # VARIOUS wins 2/3 dates → should be the final winner
    dw  = np.array([[[1], [1], [1]]], dtype=np.int32)
    dts = np.array([[[0.3], [0.3], [0.9]]], dtype=np.float32)
    winner, share = compute_abs_majority(dw, dts, n_items=2, various_idx=2)
    assert winner[0, 0] == 2


def test_abs_majority_single_date():
    """Single date, share >= 0.5 → normal winner."""
    dw  = np.array([[[1]]], dtype=np.int32)
    dts = np.array([[[0.55]]], dtype=np.float32)
    winner, share = compute_abs_majority(dw, dts, n_items=2, various_idx=2)
    assert winner[0, 0] == 1


# ── compute_weighted ───────────────────────────────────────────────────────────

def _make_weighted_data(n_bins, n_dates, n_pos, n_items):
    """Build a minimal data dict for compute_weighted tests."""
    # Create a dense winner array (all bin 0, date 0, pos 0, item 0)
    dw = np.zeros((n_bins, n_dates, n_pos), dtype=np.int32)
    # Sparse arrays: one entry per (bin, date, pos, item)
    # bin 0, date 0, pos 0: item 0 has count 7, item 1 has count 3 → item 0 wins
    wt_bin  = np.array([0, 0], dtype=np.int32)
    wt_date = np.array([0, 0], dtype=np.int32)
    wt_pos  = np.array([0, 0], dtype=np.int32)
    wt_item = np.array([0, 1], dtype=np.int32)
    wt_n    = np.array([7, 3], dtype=np.int32)
    return {
        'date_winner':    dw,
        'date_top_share': np.zeros((n_bins, n_dates, n_pos), dtype=np.float32),
        'wt_bin_idx':  wt_bin,
        'wt_date_idx': wt_date,
        'wt_pos_idx':  wt_pos,
        'wt_item_idx': wt_item,
        'wt_N_item':   wt_n,
    }


def test_weighted_basic():
    """Simple 2-item case: item 0 has count 7, item 1 has count 3 → item 0 wins."""
    data = _make_weighted_data(n_bins=1, n_dates=1, n_pos=1, n_items=2)
    winner, share, weights = compute_weighted(
        data,
        visible_bin_indices=np.array([0], dtype=np.int32),
        date_start_idx=0,
        date_end_idx=0,
        pos_indices=[0],
        n_items=2,
    )
    assert winner[0, 0] == 0
    assert abs(share[0, 0] - 0.7) < 1e-4
    assert abs(weights[0, 0, 0] - 0.7) < 1e-4
    assert abs(weights[0, 0, 1] - 0.3) < 1e-4


def test_weighted_empty():
    """No data in view → all -1."""
    data = _make_weighted_data(n_bins=2, n_dates=2, n_pos=2, n_items=2)
    # visible bin index 1 has no sparse data → empty result
    winner, share, weights = compute_weighted(
        data,
        visible_bin_indices=np.array([1], dtype=np.int32),
        date_start_idx=0,
        date_end_idx=1,
        pos_indices=[0, 1],
        n_items=2,
    )
    assert np.all(winner == -1)
    assert np.all(share == 0.0)


def test_weighted_tiebreak():
    """Tie: prefer most-recent-date winner."""
    # Two items with equal counts → tiebreak uses date_winner at date_end_idx
    n_bins, n_dates, n_pos, n_items = 1, 2, 1, 2
    dw = np.zeros((n_bins, n_dates, n_pos), dtype=np.int32)
    # Recent date (1) winner is item 1
    dw[0, 1, 0] = 1
    data = {
        'date_winner':    dw,
        'date_top_share': np.zeros((n_bins, n_dates, n_pos), dtype=np.float32),
        'wt_bin_idx':  np.array([0, 0, 0, 0], dtype=np.int32),
        'wt_date_idx': np.array([0, 0, 1, 1], dtype=np.int32),
        'wt_pos_idx':  np.array([0, 0, 0, 0], dtype=np.int32),
        'wt_item_idx': np.array([0, 1, 0, 1], dtype=np.int32),
        'wt_N_item':   np.array([5, 5, 5, 5], dtype=np.int32),  # equal counts
    }
    winner, share, weights = compute_weighted(
        data,
        visible_bin_indices=np.array([0], dtype=np.int32),
        date_start_idx=0,
        date_end_idx=1,
        pos_indices=[0],
        n_items=n_items,
    )
    assert winner[0, 0] == 1  # tiebreak: most recent date winner


# ── compute_view ───────────────────────────────────────────────────────────────

def test_compute_view_dispatches_majority():
    """compute_view with method='Majority' returns same result as compute_plurality."""
    data = generate_data()
    vis  = np.array([0, 1, 2], dtype=np.int32)
    pos  = [0, 1, 2, 3, 4]
    n_items = len(data['item_codes'])

    w1, s1, wt1 = compute_view(
        data, vis, 0, 5, pos, 'Majority', n_items, n_items,
    )
    dw_slice = data['date_winner'][vis, 0:6, :][:, :, pos]
    w2, s2   = compute_plurality(dw_slice, n_items)

    np.testing.assert_array_equal(w1, w2)
    np.testing.assert_array_almost_equal(s1, s2)
    assert wt1 is None


def test_compute_view_dispatches_abs():
    """compute_view with method='Abs. Majority' dispatches correctly."""
    data = generate_data()
    vis  = np.array([0, 1], dtype=np.int32)
    pos  = [0, 1, 2]
    n_items = len(data['item_codes'])

    w1, s1, wt1 = compute_view(
        data, vis, 0, 3, pos, 'Abs. Majority', n_items, n_items,
    )
    dw_slice  = data['date_winner'][vis, 0:4, :][:, :, pos]
    dts_slice = data['date_top_share'][vis, 0:4, :][:, :, pos]
    w2, s2    = compute_abs_majority(dw_slice, dts_slice, n_items, n_items)

    np.testing.assert_array_equal(w1, w2)
    assert wt1 is None


def test_compute_view_dispatches_weighted():
    """compute_view with method='Weighted' returns weights array (not None)."""
    data = generate_data()
    vis  = np.array([0, 1], dtype=np.int32)
    pos  = [0, 1, 2]
    n_items = len(data['item_codes'])

    w, s, wt = compute_view(
        data, vis, 0, 3, pos, 'Weighted', n_items, n_items,
    )
    assert wt is not None
    assert wt.shape == (len(vis), len(pos), n_items)
