"""Tests for generate_data and parse_uploaded_csv."""

import io
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core import (
    generate_data, ITEMS, NUM_BINS, NUM_DATES, NUM_POSITIONS,
    parse_uploaded_csv,
)

N_ITEMS = len(ITEMS)


# ── parse_uploaded_csv helpers ────────────────────────────────────────────────

def _make_csv(**kwargs) -> bytes:
    """Build a minimal valid CSV bytes object.

    Default: 3 rows with required columns. Override any column via kwargs
    (pass a list to set that column's values; pass None to omit the column).
    """
    defaults = {
        'bin_id':   ['A', 'A', 'B'],
        'date':     ['1/1/2024', '1/1/2024', '1/1/2024'],
        'position': [1, 2, 1],
        'item':     ['APX', 'BRT', 'APX'],
        'bin_rank': [1, 1, 2],
        'segment':  ['NA', 'NA', 'NA'],
    }
    defaults.update({k: v for k, v in kwargs.items() if v is not None})
    drop = [k for k, v in kwargs.items() if v is None]
    for k in drop:
        defaults.pop(k, None)
    df = pd.DataFrame(defaults)
    return df.to_csv(index=False).encode()


# ── parse_uploaded_csv tests ──────────────────────────────────────────────────

def test_parse_basic():
    """Minimal valid CSV returns a data_dict with expected keys."""
    data, msgs = parse_uploaded_csv(_make_csv())
    assert data is not None
    expected_keys = {
        'date_winner', 'date_top_share', 'wt_bin_idx', 'wt_date_idx',
        'wt_pos_idx', 'wt_item_idx', 'wt_N_item', 'bin_ranks',
        'bin_segments', 'bin_names', 'dates', 'positions', 'item_codes',
        'filter_values', '_id',
    }
    assert expected_keys.issubset(set(data.keys()))
    # No error messages
    assert not any(level == "error" for level, _ in msgs)


def test_parse_missing_column():
    """CSV missing 'item' column returns (None, [('error', ...)])."""
    data, msgs = parse_uploaded_csv(_make_csv(item=None))
    assert data is None
    assert any(level == "error" for level, _ in msgs)
    assert any("item" in text for level, text in msgs if level == "error")


def test_parse_date_format():
    """Dates in M/D/YYYY format parse correctly."""
    csv_bytes = _make_csv(date=['1/5/2024', '12/31/2024', '3/7/2024'])
    data, msgs = parse_uploaded_csv(csv_bytes)
    assert data is not None
    from datetime import date
    assert date(2024, 1, 5) in data['dates']
    assert date(2024, 12, 31) in data['dates']


def test_parse_date_fallback():
    """Unparseable dates are dropped with a warning."""
    csv_bytes = _make_csv(date=['1/1/2024', 'not-a-date', '1/8/2024'])
    data, msgs = parse_uploaded_csv(csv_bytes)
    # Should warn about dropped row
    assert any(level == "warning" and "unparseable dates" in text for level, text in msgs)
    # Data should still be returned (2 valid rows remain)
    assert data is not None


def test_parse_N_item_default():
    """CSV without N_item column works; each row counts as 1."""
    csv_bytes = _make_csv()
    assert b'N_item' not in csv_bytes
    data, msgs = parse_uploaded_csv(csv_bytes)
    assert data is not None
    # All wt_N_item values should be 1
    assert np.all(data['wt_N_item'] == 1)


def test_parse_N_item_explicit():
    """N_item values are summed for duplicate keys."""
    # Two rows for same (bin_id, date, position, item, segment) with N_item=3 and 5
    df = pd.DataFrame({
        'bin_id':   ['A', 'A'],
        'date':     ['1/1/2024', '1/1/2024'],
        'position': [1, 1],
        'item':     ['APX', 'APX'],
        'bin_rank': [1, 1],
        'segment':  ['NA', 'NA'],
        'N_item':   [3, 5],
    })
    data, msgs = parse_uploaded_csv(df.to_csv(index=False).encode())
    assert data is not None
    # Summed N_item = 8
    assert data['wt_N_item'].sum() == 8


def test_parse_deduplication():
    """Duplicate (bin_id, date, position, item, segment) rows are summed."""
    df = pd.DataFrame({
        'bin_id':   ['A', 'A', 'A'],
        'date':     ['1/1/2024', '1/1/2024', '1/1/2024'],
        'position': [1, 1, 2],
        'item':     ['APX', 'APX', 'BRT'],
        'bin_rank': [1, 1, 1],
        'segment':  ['NA', 'NA', 'NA'],
    })
    data, msgs = parse_uploaded_csv(df.to_csv(index=False).encode())
    assert data is not None
    # After dedup: 2 distinct (bin, date, pos, item) groups
    assert len(data['wt_N_item']) == 2
    # The APX entry should have count 2 (2 rows summed)
    assert data['wt_N_item'].max() == 2


def test_parse_filter_column():
    """CSV with 'filter' column produces date_winner_by_filter not None."""
    df = pd.DataFrame({
        'bin_id':   ['A', 'A'],
        'date':     ['1/1/2024', '1/1/2024'],
        'position': [1, 1],
        'item':     ['APX', 'BRT'],
        'bin_rank': [1, 1],
        'segment':  ['NA', 'NA'],
        'filter':   ['F1', 'F2'],
    })
    data, msgs = parse_uploaded_csv(df.to_csv(index=False).encode())
    assert data is not None
    assert data['date_winner_by_filter'] is not None
    assert data['date_winner'] is None


def test_parse_fractional_position():
    """Rows with fractional position are dropped with a warning."""
    df = pd.DataFrame({
        'bin_id':   ['A', 'A'],
        'date':     ['1/1/2024', '1/1/2024'],
        'position': [1, 1.5],
        'item':     ['APX', 'BRT'],
        'bin_rank': [1, 1],
        'segment':  ['NA', 'NA'],
    })
    data, msgs = parse_uploaded_csv(df.to_csv(index=False).encode())
    assert any("fractional position" in text for _, text in msgs)
    # 1 valid row remains
    assert data is not None
    assert len(data['positions']) == 1


def test_parse_empty_after_cleaning():
    """If all rows are dropped, returns (None, [('error', ...)])."""
    # All rows have fractional positions → all dropped
    df = pd.DataFrame({
        'bin_id':   ['A'],
        'date':     ['1/1/2024'],
        'position': [1.5],
        'item':     ['APX'],
        'bin_rank': [1],
        'segment':  ['NA'],
    })
    data, msgs = parse_uploaded_csv(df.to_csv(index=False).encode())
    assert data is None
    assert any(level == "error" for level, _ in msgs)


def test_parse_multi_segment_per_bin():
    """Same bin_id with two different segments → two distinct bin_keys."""
    df = pd.DataFrame({
        'bin_id':   ['A', 'A'],
        'date':     ['1/1/2024', '1/1/2024'],
        'position': [1, 1],
        'item':     ['APX', 'APX'],
        'bin_rank': [1, 1],
        'segment':  ['NA', 'EU'],
    })
    data, msgs = parse_uploaded_csv(df.to_csv(index=False).encode())
    assert data is not None
    assert len(data['bin_names']) == 2
    assert any('NA' in name for name in data['bin_names'])
    assert any('EU' in name for name in data['bin_names'])


def test_parse_winner_range():
    """All winners in [-1, n_items-1]."""
    csv_bytes = _make_csv(
        bin_id=['A', 'A', 'B', 'B'],
        date=['1/1/2024', '1/8/2024', '1/1/2024', '1/8/2024'],
        position=[1, 1, 1, 1],
        item=['APX', 'BRT', 'APX', 'BRT'],
        bin_rank=[1, 1, 2, 2],
        segment=['NA', 'NA', 'NA', 'NA'],
    )
    data, msgs = parse_uploaded_csv(csv_bytes)
    assert data is not None
    n_items = len(data['item_codes'])
    dw = data['date_winner']
    assert dw.min() >= -1
    assert dw.max() <= n_items - 1


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
