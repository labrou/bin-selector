"""
Performance benchmarks for core compute functions.

Run with:
    pytest tests/test_performance.py -v --benchmark-sort=mean
    pytest tests/test_performance.py -v --benchmark-histogram   (ASCII chart)
    pytest tests/test_performance.py -v --benchmark-disable     (skip timing, just verify correctness)
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core import (
    compute_plurality,
    compute_abs_majority,
    compute_weighted,
    compute_view,
    compute_sort_order,
    generate_data,
    parse_uploaded_csv,
    make_view_csv,
    dim_color,
    ITEMS,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def full_data():
    """Full synthetic dataset (100 bins × 52 dates × 50 positions)."""
    return generate_data()


@pytest.fixture(scope="module")
def small_winner(full_data):
    """Slice covering 20 bins × 13 dates × 50 positions — typical filtered view."""
    dw = full_data["date_winner"][:20, 39:52, :]   # last 13 dates
    ds = full_data["date_top_share"][:20, 39:52, :]
    return dw, ds


@pytest.fixture(scope="module")
def large_winner(full_data):
    """Full slice: 100 bins × 52 dates × 50 positions."""
    return full_data["date_winner"], full_data["date_top_share"]


@pytest.fixture(scope="module")
def medium_csv_bytes():
    """~600-row CSV: 10 bins × 6 dates × 10 positions × 1 item each."""
    import io, datetime
    rows = ["bin_id,date,position,item,bin_rank,segment"]
    base = datetime.date(2024, 1, 1)
    items = ["APX", "BRT", "CFD"]
    for b in range(10):
        for d in range(6):
            dt = base + datetime.timedelta(weeks=d)
            for p in range(1, 11):
                item = items[(b + p) % 3]
                rows.append(f"Bin{b:02d},{dt.month}/{dt.day}/{dt.year},{p},{item},{b+1},NA")
    return "\n".join(rows).encode()


@pytest.fixture(scope="module")
def large_csv_bytes():
    """~5 000-row CSV: 20 bins × 25 dates × 10 positions × 1 item."""
    import io, datetime
    rows = ["bin_id,date,position,item,bin_rank,segment"]
    base = datetime.date(2024, 1, 1)
    items = ["APX", "BRT", "CFD", "DLT", "ETR"]
    for b in range(20):
        for d in range(25):
            dt = base + datetime.timedelta(weeks=d)
            for p in range(1, 11):
                item = items[(b + p) % 5]
                rows.append(f"Bin{b:02d},{dt.month}/{dt.day}/{dt.year},{p},{item},{b+1},NA")
    return "\n".join(rows).encode()


# ── compute_plurality ─────────────────────────────────────────────────────────

def test_bench_plurality_small(benchmark, small_winner):
    dw, _ = small_winner
    n_items = len(ITEMS)
    result = benchmark(compute_plurality, dw, n_items)
    winner, share = result
    assert winner.shape == (20, 50)


def test_bench_plurality_full(benchmark, large_winner):
    dw, _ = large_winner
    n_items = len(ITEMS)
    result = benchmark(compute_plurality, dw, n_items)
    winner, share = result
    assert winner.shape == (100, 50)


# ── compute_abs_majority ──────────────────────────────────────────────────────

def test_bench_abs_majority_small(benchmark, small_winner):
    dw, ds = small_winner
    n_items = len(ITEMS)
    various_idx = n_items
    result = benchmark(compute_abs_majority, dw, ds, n_items, various_idx)
    winner, share = result
    assert winner.shape == (20, 50)


def test_bench_abs_majority_full(benchmark, large_winner):
    dw, ds = large_winner
    n_items = len(ITEMS)
    various_idx = n_items
    result = benchmark(compute_abs_majority, dw, ds, n_items, various_idx)
    winner, share = result
    assert winner.shape == (100, 50)


# ── compute_weighted ──────────────────────────────────────────────────────────

def test_bench_weighted_small(benchmark, full_data):
    n_items = len(ITEMS)
    vis = np.arange(20)
    pos = list(range(50))
    result = benchmark(
        compute_weighted, full_data, vis, 39, 51, pos, n_items
    )
    winner, share, weights = result
    assert winner.shape == (20, 50)


def test_bench_weighted_full(benchmark, full_data):
    n_items = len(ITEMS)
    vis = np.arange(100)
    pos = list(range(50))
    result = benchmark(
        compute_weighted, full_data, vis, 0, 51, pos, n_items
    )
    winner, share, weights = result
    assert winner.shape == (100, 50)


# ── compute_view (dispatch) ───────────────────────────────────────────────────

@pytest.mark.parametrize("method", ["Majority", "Abs. Majority", "Weighted"])
def test_bench_compute_view(benchmark, full_data, method):
    n_items = len(ITEMS)
    various_idx = n_items
    vis = np.arange(100)
    pos = list(range(50))
    result = benchmark(
        compute_view,
        full_data, vis, 0, 51, pos, method, n_items, various_idx,
    )
    winner, share, weights = result
    assert winner.shape == (100, 50)


# ── compute_sort_order ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def majority_view_for_sort(full_data):
    dw = full_data["date_winner"][:100, 39:52, :]
    winner, _ = compute_plurality(dw, len(ITEMS))
    return winner


@pytest.mark.parametrize("sort_mode", ["Index", "Similarity", "Bin Rank", "Top-rank"])
def test_bench_sort(benchmark, majority_view_for_sort, full_data, sort_mode):
    bin_ranks = full_data["bin_ranks"][:100]
    n_items = len(ITEMS)
    pos_indices = list(range(50))
    result = benchmark(
        compute_sort_order,
        sort_mode, majority_view_for_sort, bin_ranks, n_items,
        [], ITEMS, pos_indices, len(ITEMS),
    )
    assert len(result) == 100


# ── parse_uploaded_csv ────────────────────────────────────────────────────────

def test_bench_parse_medium_csv(benchmark, medium_csv_bytes):
    result = benchmark(parse_uploaded_csv, medium_csv_bytes)
    data_dict, messages = result
    assert data_dict is not None


def test_bench_parse_large_csv(benchmark, large_csv_bytes):
    result = benchmark(parse_uploaded_csv, large_csv_bytes)
    data_dict, messages = result
    assert data_dict is not None


# ── make_view_csv ─────────────────────────────────────────────────────────────

def test_bench_make_view_csv(benchmark, full_data):
    dw = full_data["date_winner"][:100, 39:52, :]
    winner, share = compute_plurality(dw, len(ITEMS))
    bin_names = full_data["bin_names"][:100]
    positions = np.array(full_data["positions"])
    ranks = full_data["bin_ranks"][:100]
    segments = full_data["bin_segments"][:100]
    result = benchmark(
        make_view_csv,
        bin_names, positions, winner, share, ranks, segments,
        item_codes=ITEMS, bin_term="bin", method="Majority",
    )
    assert len(result) > 0


# ── dim_color (micro-benchmark) ───────────────────────────────────────────────

def test_bench_dim_color(benchmark):
    result = benchmark(dim_color, "#B91C1C", 0.88, "#F7F4ED")
    assert result.startswith("#")
