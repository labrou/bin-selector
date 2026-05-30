"""End-to-end integration tests: parse_uploaded_csv → compute_view → make_view_csv."""

import io
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core import (
    parse_uploaded_csv, compute_view, make_view_csv, compute_sort_order,
)

# ── Shared fixture CSV ────────────────────────────────────────────────────────
FIXTURE_CSV = b"""bin_id,date,position,item,bin_rank,segment
Alpha,1/1/2024,1,APX,1,NA
Alpha,1/1/2024,2,BRT,1,NA
Alpha,1/1/2024,3,APX,1,NA
Alpha,1/8/2024,1,APX,1,NA
Alpha,1/8/2024,2,APX,1,NA
Alpha,1/8/2024,3,BRT,1,NA
Beta,1/1/2024,1,BRT,2,NA
Beta,1/1/2024,2,APX,2,NA
Beta,1/1/2024,3,BRT,2,NA
Beta,1/8/2024,1,BRT,2,NA
Beta,1/8/2024,2,BRT,2,NA
Beta,1/8/2024,3,APX,2,NA
Gamma,1/1/2024,1,APX,3,EU
Gamma,1/1/2024,2,APX,3,EU
Gamma,1/1/2024,3,APX,3,EU
Gamma,1/8/2024,1,BRT,3,EU
Gamma,1/8/2024,2,BRT,3,EU
Gamma,1/8/2024,3,BRT,3,EU
"""


@pytest.fixture(scope="module")
def parsed():
    data, msgs = parse_uploaded_csv(FIXTURE_CSV)
    assert data is not None, f"Parse failed: {msgs}"
    return data


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_integration_parse_shape(parsed):
    """parse returns 3 bins, 2 dates, 3 positions, 2 items."""
    assert len(parsed['bin_names']) == 3
    assert len(parsed['dates']) == 2
    assert len(parsed['positions']) == 3
    assert len(parsed['item_codes']) == 2


def test_integration_majority_winner(parsed):
    """Alpha P1: APX wins 2/2 dates → winner=0 (APX index, highest freq item)."""
    item_codes = parsed['item_codes']
    apx_idx = item_codes.index('APX')
    # Alpha is 'Alpha · NA'
    alpha_idx = list(parsed['bin_names']).index('Alpha · NA')
    n_items = len(item_codes)
    various_idx = n_items

    winner, share, _ = compute_view(
        parsed,
        visible_bin_indices=np.array([alpha_idx], dtype=np.int32),
        date_start_idx=0,
        date_end_idx=1,
        pos_indices=[0],      # position index 0 = position 1
        method='Majority',
        n_items=n_items,
        various_idx=various_idx,
    )
    # Alpha P1: APX on both dates → plurality winner = APX
    assert winner[0, 0] == apx_idx


def test_integration_majority_tie(parsed):
    """Alpha P2: BRT date1, APX date2 → tiebreak = most recent (APX)."""
    item_codes = parsed['item_codes']
    apx_idx = item_codes.index('APX')
    alpha_idx = list(parsed['bin_names']).index('Alpha · NA')
    n_items = len(item_codes)

    winner, share, _ = compute_view(
        parsed,
        visible_bin_indices=np.array([alpha_idx], dtype=np.int32),
        date_start_idx=0,
        date_end_idx=1,
        pos_indices=[1],      # position index 1 = position 2
        method='Majority',
        n_items=n_items,
        various_idx=n_items,
    )
    # Alpha P2: date1=BRT, date2=APX → tie, tiebreak = most recent = APX
    assert winner[0, 0] == apx_idx


def test_integration_abs_majority_various(parsed):
    """Abs. Majority with clear winners at P1: no VARIOUS at Alpha P1."""
    item_codes = parsed['item_codes']
    n_items = len(item_codes)
    various_idx = n_items
    alpha_idx = list(parsed['bin_names']).index('Alpha · NA')

    winner, share, _ = compute_view(
        parsed,
        visible_bin_indices=np.array([alpha_idx], dtype=np.int32),
        date_start_idx=0,
        date_end_idx=1,
        pos_indices=[0],
        method='Abs. Majority',
        n_items=n_items,
        various_idx=various_idx,
    )
    # Alpha P1: APX wins both dates with 100% share each → NOT VARIOUS
    assert winner[0, 0] != various_idx


def test_integration_weighted_share(parsed):
    """Alpha P1: 2 APX observations total → APX wins with highest share."""
    item_codes = parsed['item_codes']
    apx_idx = item_codes.index('APX')
    n_items = len(item_codes)
    alpha_idx = list(parsed['bin_names']).index('Alpha · NA')

    winner, share, weights = compute_view(
        parsed,
        visible_bin_indices=np.array([alpha_idx], dtype=np.int32),
        date_start_idx=0,
        date_end_idx=1,
        pos_indices=[0],
        method='Weighted',
        n_items=n_items,
        various_idx=n_items,
    )
    assert winner[0, 0] == apx_idx
    # APX has 2/2 observations at P1 for Alpha → share = 1.0
    assert abs(float(weights[0, 0, apx_idx]) - 1.0) < 1e-4


def test_integration_compute_view_all_methods(parsed):
    """compute_view with each method returns correct shapes."""
    n_items = len(parsed['item_codes'])
    vis = np.arange(len(parsed['bin_names']), dtype=np.int32)
    pos = list(range(len(parsed['positions'])))

    for method in ['Majority', 'Abs. Majority', 'Weighted']:
        winner, share, weights = compute_view(
            parsed, vis, 0, 1, pos, method, n_items, n_items,
        )
        assert winner.shape == (len(vis), len(pos)), f"Wrong shape for {method}"
        assert share.shape == (len(vis), len(pos)), f"Wrong share shape for {method}"
        if method == 'Weighted':
            assert weights is not None
            assert weights.shape == (len(vis), len(pos), n_items)
        else:
            assert weights is None


def test_integration_make_view_csv_roundtrip(parsed):
    """make_view_csv output loads as DataFrame with expected columns and row count."""
    n_items = len(parsed['item_codes'])
    vis = np.arange(len(parsed['bin_names']), dtype=np.int32)
    pos = list(range(len(parsed['positions'])))

    winner, share, _ = compute_view(
        parsed, vis, 0, 1, pos, 'Majority', n_items, n_items,
    )
    csv_bytes = make_view_csv(
        parsed['bin_names'],
        np.array(parsed['positions']),
        winner,
        share,
        parsed['bin_ranks'],
        parsed['bin_segments'],
        item_codes=parsed['item_codes'],
        bin_term='bin',
        method='Majority',
    )
    df = pd.read_csv(io.BytesIO(csv_bytes))
    assert 'bin' in df.columns
    assert 'item' in df.columns
    assert len(df) == len(parsed['bin_names']) * len(parsed['positions'])


def test_integration_segment_filter(parsed):
    """visible_bin_indices filtered to NA gives Alpha and Beta only."""
    bin_names = parsed['bin_names']
    segments  = parsed['bin_segments']
    na_mask   = segments == 'NA'
    na_indices = np.where(na_mask)[0]
    na_names   = bin_names[na_indices]
    assert len(na_indices) == 2
    assert any('Alpha' in n for n in na_names)
    assert any('Beta' in n for n in na_names)
    assert not any('Gamma' in n for n in na_names)


def test_integration_sort_matches_expected(parsed):
    """compute_sort_order(Index) returns [0,1,...,n-1]."""
    n_items = len(parsed['item_codes'])
    vis = np.arange(len(parsed['bin_names']), dtype=np.int32)
    pos = list(range(len(parsed['positions'])))

    winner, share, _ = compute_view(
        parsed, vis, 0, 1, pos, 'Majority', n_items, n_items,
    )
    order = compute_sort_order(
        "Index", winner, parsed['bin_ranks'][vis], n_items,
        [], parsed['item_codes'], pos, n_items,
    )
    np.testing.assert_array_equal(order, np.arange(len(vis)))
