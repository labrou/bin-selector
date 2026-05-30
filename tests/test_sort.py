"""Tests for compute_sort_order."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core import compute_sort_order

# ── Shared test fixtures ───────────────────────────────────────────────────────
# 5 bins, 4 positions, 3 items (0=Red, 1=Blue, 2=Green)
MAJORITY = np.array([
    [0, 0, 0, 1],  # bin 0: Red-dominant
    [0, 0, 1, 2],  # bin 1: Red then mixed
    [1, 1, 0, 0],  # bin 2: Blue-dominant
    [2, 2, 2, 0],  # bin 3: Green-dominant
    [0, 1, 0, 1],  # bin 4: alternating
], dtype=np.int32)
BIN_RANKS = np.array([3, 5, 2, 1, 4])
ITEM_CODES = ["Red", "Blue", "Green"]
POS_INDICES = [0, 1, 2, 3]
N_PILL_ITEMS = 3
N_ITEMS = 3


def _sort(sort_mode, selected_items=None):
    return compute_sort_order(
        sort_mode, MAJORITY, BIN_RANKS, N_ITEMS,
        selected_items or [], ITEM_CODES, POS_INDICES, N_PILL_ITEMS,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_sort_index():
    """Index sort returns [0,1,2,3,4] (unchanged)."""
    order = _sort("Index")
    np.testing.assert_array_equal(order, [0, 1, 2, 3, 4])


def test_sort_bin_rank():
    """Bin Rank sort returns order by ascending rank: [3,2,0,4,1] (ranks 1,2,3,4,5)."""
    # BIN_RANKS = [3,5,2,1,4]; ascending → indices [3,2,0,4,1]
    order = _sort("Bin Rank")
    np.testing.assert_array_equal(order, [3, 2, 0, 4, 1])


def test_sort_similarity():
    """Similarity sort: bins sharing P1 winner are grouped before others."""
    order = _sort("Similarity")
    # Should be a valid permutation of 5 elements
    assert len(order) == 5
    assert set(order.tolist()) == {0, 1, 2, 3, 4}
    # Bins 0, 1, 4 all have Red (0) at P1; bins 2 and 3 do not.
    # After lexsort on top4, the Red-P1 group occupies consecutive positions.
    red_p1_bins = {0, 1, 4}
    positions_of_red_group = sorted(
        int(np.where(order == b)[0][0]) for b in red_p1_bins
    )
    # The three positions should be contiguous
    assert positions_of_red_group == list(range(
        positions_of_red_group[0], positions_of_red_group[0] + 3
    ))


def test_sort_top_rank():
    """Top-rank sort: groups by P1 winner, resolves ties by P2, P3..."""
    order = _sort("Top-rank")
    assert len(order) == 5
    assert set(order.tolist()) == {0, 1, 2, 3, 4}
    # Bins with Red at P1: 0,1,4 (values 0); Blue at P1: 2 (value 1); Green: 3 (value 2)
    # lexsort sorts ascending by last key first → Red(0) first, then Blue(1), then Green(2)
    # Find positions of bins 2 and 3 — bin 2 (Blue P1) should come before bin 3 (Green P1)
    pos_2 = int(np.where(order == 2)[0][0])
    pos_3 = int(np.where(order == 3)[0][0])
    assert pos_2 < pos_3


def test_sort_selected_share_one_item():
    """Select 'Red': bins ranked by Red count descending."""
    order = _sort("Selected Share", selected_items=["Red"])
    assert len(order) == 5
    # Count Red occurrences per bin in MAJORITY
    red_counts = (MAJORITY == 0).sum(axis=1)  # [3, 2, 2, 1, 2]
    # The first bin in order should have the most Red
    assert red_counts[order[0]] == red_counts.max()


def test_sort_selected_share_tiebreak():
    """Two bins with same Red count: one with Red at P1 ranks higher (lower pos_sum)."""
    # Bins 1 and 4 both have 2 Red positions each
    # Bin 1: Red at positions 0,1 → pos_sum = 0+1+1=2 (indices), lower → ranked higher
    # Bin 4: Red at positions 0,2 → pos_sum = 0+1+3=4 ... let's verify by computation
    order = _sort("Selected Share", selected_items=["Red"])
    # Bins 1 and 4 have same count (2 each) but different positions
    # Bin 1: positions 0,1 → pos_sum = (0+1)*1 = 1 (indices 0+1=1, +1 each)
    # Actual pos_sum = (sel_mask * (pos_indices + 1)).sum(axis=1)
    # pos_indices=[0,1,2,3], so weights=[1,2,3,4]
    # Bin 1: Red at pos 0,1 → 1+2=3
    # Bin 4: Red at pos 0,2 → 1+3=4
    # Lower pos_sum → ranked higher → bin 1 before bin 4
    pos_1 = int(np.where(order == 1)[0][0])
    pos_4 = int(np.where(order == 4)[0][0])
    assert pos_1 < pos_4


def test_sort_returns_array_of_correct_shape():
    """Output length == n_vis."""
    for mode in ["Index", "Similarity", "Top-rank", "Bin Rank"]:
        order = _sort(mode)
        assert len(order) == MAJORITY.shape[0], f"Wrong length for {mode}"


def test_sort_bin_rank_mode_name_variant():
    """'Store Rank' works the same as 'Bin Rank' (ends with ' Rank')."""
    order_bin   = _sort("Bin Rank")
    order_store = compute_sort_order(
        "Store Rank", MAJORITY, BIN_RANKS, N_ITEMS,
        [], ITEM_CODES, POS_INDICES, N_PILL_ITEMS,
    )
    np.testing.assert_array_equal(order_bin, order_store)
