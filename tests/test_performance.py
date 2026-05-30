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


# ── Large-scale fixtures ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def large_synthetic():
    """200 bins × 52 dates × 100 positions × 1000 items × 10 segments.

    Builds compact arrays (date_winner, date_top_share) and sparse long arrays
    (wt_*) manually using numpy, without calling generate_data().

    Sparse density: ~3-5 observations per (bin, date, position) cell on average.
    1000 items stress-tests the bincount/reshape operations in all compute paths.
    """
    rng = np.random.default_rng(99)
    N_BINS     = 200
    N_DATES    = 52
    N_POS      = 100
    N_ITEMS    = 1000
    N_SEGMENTS = 10

    # Per-cell observation counts: Poisson(4) clipped to [1, 10]
    k_obs = rng.integers(1, 11, (N_BINS, N_DATES, N_POS), dtype=np.int32)

    # Uniform item probabilities (simple — we only care about timing)
    probs = np.ones(N_ITEMS, dtype=np.float64) / N_ITEMS

    # Compact arrays
    date_winner    = np.full((N_BINS, N_DATES, N_POS), -1,         dtype=np.int32)
    date_top_share = np.zeros((N_BINS, N_DATES, N_POS),            dtype=np.float32)

    # Sparse long arrays (built per-bin to avoid a huge 500×52×100×20 dense cube)
    wt_bins  = []
    wt_dates = []
    wt_pos_l = []
    wt_items = []
    wt_ni    = []

    for b in range(N_BINS):
        k_b   = k_obs[b]            # (N_DATES, N_POS)
        max_k = int(k_b.max())

        # Draw item indices for every observation
        r = rng.integers(0, N_ITEMS, (N_DATES, N_POS, max_k), dtype=np.int32)
        valid_mask = np.arange(max_k)[None, None, :] < k_b[:, :, None]
        d_idx, p_idx, k_idx = np.where(valid_mask)
        item_draws = r[d_idx, p_idx, k_idx].astype(np.intp)

        counts_b = np.zeros((N_DATES, N_POS, N_ITEMS), dtype=np.int32)
        np.add.at(counts_b, (d_idx, p_idx, item_draws), 1)

        group_n_b   = counts_b.sum(axis=-1)
        has_data_b  = group_n_b > 0
        argmax_b    = counts_b.argmax(axis=-1)
        max_cnt_b   = counts_b.max(axis=-1)

        date_winner[b]    = np.where(has_data_b, argmax_b, -1).astype(np.int32)
        date_top_share[b] = np.where(
            has_data_b,
            (max_cnt_b / np.maximum(group_n_b, 1)).astype(np.float32),
            np.float32(0),
        )

        di_nz, pi_nz, ii_nz = np.where(counts_b > 0)
        if len(di_nz) > 0:
            wt_bins.append(np.full(len(di_nz), b, dtype=np.int32))
            wt_dates.append(di_nz.astype(np.int32))
            wt_pos_l.append(pi_nz.astype(np.int32))
            wt_items.append(ii_nz.astype(np.int32))
            wt_ni.append(counts_b[di_nz, pi_nz, ii_nz].astype(np.int32))

    _cat = lambda lst, dt: np.concatenate(lst) if lst else np.array([], dtype=dt)
    bin_ranks    = rng.integers(1, N_BINS + 1, N_BINS, dtype=np.int32)
    seg_labels   = [f'SEG{s:02d}' for s in range(N_SEGMENTS)]
    bin_segments = np.array([seg_labels[b % N_SEGMENTS] for b in range(N_BINS)])
    bin_names    = np.array([f'Bin{b:04d}' for b in range(N_BINS)])

    import datetime
    end_date = datetime.date(2025, 1, 1)
    dates    = [end_date - datetime.timedelta(weeks=N_DATES - 1 - w) for w in range(N_DATES)]

    return {
        'date_winner':              date_winner,
        'date_top_share':           date_top_share,
        'date_winner_by_filter':    None,
        'date_top_share_by_filter': None,
        'wt_bin_idx':    _cat(wt_bins,  np.int32),
        'wt_date_idx':   _cat(wt_dates, np.int32),
        'wt_pos_idx':    _cat(wt_pos_l, np.int32),
        'wt_item_idx':   _cat(wt_items, np.int32),
        'wt_N_item':     _cat(wt_ni,    np.int32),
        'wt_filter_idx': None,
        'bin_ranks':     bin_ranks,
        'bin_segments':  bin_segments,
        'bin_names':     bin_names,
        'dates':         dates,
        'positions':     list(range(1, N_POS + 1)),
        'item_codes':    [f'ITEM{i:04d}' for i in range(N_ITEMS)],
        '_id':           'large_synthetic',
        '_n_bins':       N_BINS,
        '_n_dates':      N_DATES,
        '_n_pos':        N_POS,
        '_n_items':      N_ITEMS,
    }


# ── Large-scale benchmarks ────────────────────────────────────────────────────

def test_bench_plurality_large_scale(benchmark, large_synthetic):
    """200 bins × 52 dates × 100 positions × 1000 items — Majority."""
    d = large_synthetic
    dw = d['date_winner']  # (500, 52, 100)
    n_items = d['_n_items']
    result = benchmark(compute_plurality, dw, n_items)
    winner, share = result
    assert winner.shape == (d['_n_bins'], d['_n_pos'])


def test_bench_abs_majority_large_scale(benchmark, large_synthetic):
    """200 bins × 52 dates × 100 positions × 1000 items — Abs. Majority."""
    d = large_synthetic
    dw  = d['date_winner']
    dts = d['date_top_share']
    n_items = d['_n_items']
    various_idx = n_items
    result = benchmark(compute_abs_majority, dw, dts, n_items, various_idx)
    winner, share = result
    assert winner.shape == (d['_n_bins'], d['_n_pos'])


def test_bench_weighted_large_scale(benchmark, large_synthetic):
    """200 bins × 52 dates × 100 positions × 1000 items — Weighted."""
    d = large_synthetic
    n_items = d['_n_items']
    vis = np.arange(d['_n_bins'])
    pos = list(range(d['_n_pos']))
    result = benchmark(
        compute_weighted, d, vis, 0, d['_n_dates'] - 1, pos, n_items
    )
    winner, share, weights = result
    assert winner.shape == (d['_n_bins'], d['_n_pos'])


def test_bench_compute_view_weighted_large_scale(benchmark, large_synthetic):
    """compute_view dispatch — Weighted, 200 bins × 52 dates × 100 positions × 1000 items."""
    d = large_synthetic
    n_items = d['_n_items']
    various_idx = n_items
    vis = np.arange(d['_n_bins'])
    pos = list(range(d['_n_pos']))
    result = benchmark(
        compute_view,
        d, vis, 0, d['_n_dates'] - 1, pos, 'Weighted', n_items, various_idx,
    )
    winner, share, weights = result
    assert winner.shape == (d['_n_bins'], d['_n_pos'])


@pytest.fixture(scope="module")
def very_large_csv_bytes():
    """100 bins × 52 dates × 20 positions × 3 items = ~312 000 rows."""
    import datetime
    rows = ["bin_id,date,position,item,bin_rank,segment"]
    base  = datetime.date(2024, 1, 1)
    items = ["APX", "BRT", "CFD"]
    for b in range(100):
        for d in range(52):
            dt = base + datetime.timedelta(weeks=d)
            for p in range(1, 21):
                for item in items:
                    rows.append(
                        f"Bin{b:03d},{dt.month}/{dt.day}/{dt.year},{p},{item},{b+1},NA"
                    )
    return "\n".join(rows).encode()


def test_bench_parse_very_large_csv(benchmark, very_large_csv_bytes):
    """parse_uploaded_csv on ~312 000-row CSV."""
    result = benchmark(parse_uploaded_csv, very_large_csv_bytes)
    data_dict, messages = result
    assert data_dict is not None
