"""Tests for dim_color and make_view_csv helper functions."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core import dim_color, make_view_csv, BG, VARIOUS_LABEL, ITEMS


# ── dim_color ──────────────────────────────────────────────────────────────────

def test_dim_color_full_dim():
    """dim_amount=1.0 → output equals background."""
    fg = '#B91C1C'
    result = dim_color(fg, dim_amount=1.0, bg=BG)
    assert result.lower() == BG.lower()


def test_dim_color_no_dim():
    """dim_amount=0.0 → output equals foreground (hex-normalised)."""
    fg = '#B91C1C'
    result = dim_color(fg, dim_amount=0.0, bg=BG)
    # Compare channels because the function normalises to lowercase 2-digit hex
    fg_channels  = tuple(int(fg[i:i+2], 16) for i in (1, 3, 5))
    out_channels = tuple(int(result[i:i+2], 16) for i in (1, 3, 5))
    assert fg_channels == out_channels


def test_dim_color_midpoint():
    """dim_amount=0.5 → each channel is midpoint between fg and bg."""
    fg = '#B91C1C'
    result = dim_color(fg, dim_amount=0.5, bg=BG)
    fg_ch  = [int(fg[i:i+2], 16) for i in (1, 3, 5)]
    bg_ch  = [int(BG[i:i+2], 16) for i in (1, 3, 5)]
    res_ch = [int(result[i:i+2], 16) for i in (1, 3, 5)]
    for j in range(3):
        expected = int(fg_ch[j] * 0.5 + bg_ch[j] * 0.5)
        assert res_ch[j] == expected


def test_dim_color_format():
    """Output is 7-char hex string starting with #."""
    result = dim_color('#B91C1C', dim_amount=0.5, bg=BG)
    assert isinstance(result, str)
    assert len(result) == 7
    assert result.startswith('#')
    # All chars after # are valid hex
    int(result[1:], 16)  # raises ValueError if not hex


# ── make_view_csv ──────────────────────────────────────────────────────────────

def _make_grid(n_bins=3, n_pos=4, n_items=5):
    """Return a simple items_grid and share_grid for CSV tests."""
    rng = np.random.default_rng(0)
    items_grid = rng.integers(0, n_items, (n_bins, n_pos), dtype=np.int32)
    share_grid = rng.random((n_bins, n_pos)).astype(np.float32)
    return items_grid, share_grid


def test_make_view_csv_columns():
    """CSV has expected columns for Majority method."""
    n_bins, n_pos = 3, 4
    items_grid, share_grid = _make_grid(n_bins, n_pos, n_items=5)
    item_codes = ITEMS[:5]
    bin_names  = np.array([f'Bin{i}' for i in range(n_bins)])
    positions  = np.arange(1, n_pos + 1)
    ranks      = np.arange(n_bins)
    segments   = np.array(['NA'] * n_bins)

    csv_bytes = make_view_csv(
        bin_names, positions, items_grid, share_grid, ranks, segments,
        item_codes=item_codes, bin_term='bin', method='Majority',
    )
    import io, pandas as pd
    df = pd.read_csv(io.BytesIO(csv_bytes))
    assert 'bin' in df.columns
    assert 'rank' in df.columns
    assert 'segment' in df.columns
    assert 'position' in df.columns
    assert 'item' in df.columns
    assert 'item_share' in df.columns


def test_make_view_csv_shape():
    """Correct number of rows: n_bins * n_positions."""
    n_bins, n_pos = 5, 6
    items_grid, share_grid = _make_grid(n_bins, n_pos, n_items=3)
    bin_names  = np.array([f'B{i}' for i in range(n_bins)])
    positions  = np.arange(1, n_pos + 1)
    ranks      = np.zeros(n_bins, dtype=int)
    segments   = np.array(['NA'] * n_bins)

    csv_bytes = make_view_csv(
        bin_names, positions, items_grid, share_grid, ranks, segments,
        item_codes=ITEMS[:3], bin_term='bin', method='Majority',
    )
    import io, pandas as pd
    df = pd.read_csv(io.BytesIO(csv_bytes))
    assert len(df) == n_bins * n_pos


def test_make_view_csv_weighted_extra_cols():
    """Weighted method adds share_<item> columns for colored items."""
    n_bins, n_pos, n_items = 2, 3, 4
    items_grid, share_grid = _make_grid(n_bins, n_pos, n_items)
    item_codes = ITEMS[:n_items]
    bin_names  = np.array([f'B{i}' for i in range(n_bins)])
    positions  = np.arange(1, n_pos + 1)
    ranks      = np.zeros(n_bins, dtype=int)
    segments   = np.array(['NA'] * n_bins)
    weights_grid = np.random.rand(n_bins, n_pos, n_items).astype(np.float32)

    csv_bytes = make_view_csv(
        bin_names, positions, items_grid, share_grid, ranks, segments,
        item_codes=item_codes, bin_term='bin', method='Weighted',
        weights_grid=weights_grid, colored_item_codes=item_codes,
    )
    import io, pandas as pd
    df = pd.read_csv(io.BytesIO(csv_bytes))
    # Should have weighted_share and share_APX, share_BRT, share_CFD, share_DLT
    assert 'weighted_share' in df.columns
    for code in item_codes:
        assert f'share_{code}' in df.columns
