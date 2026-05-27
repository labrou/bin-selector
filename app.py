"""
Ranked Placement Atlas - Streamlit version
==========================================

Run:
    pip install streamlit>=1.40 plotly numpy pandas kaleido
    streamlit run app.py

Features:
  - Categorical heatmap: bins × positions × items × snapshots
  - Region / rank / position / date filters with shareable URL state
  - Sort modes: Index, Similarity, <bin_term> Rank, Top-rank, Selected Share
  - Aggregation methods: Majority, Abs. Majority, Weighted
  - Item highlight (dims non-selected items)
  - Legend mapping item codes to colors; VARIOUS shown as a distinct colour
  - Click any bin row to open its full time-series heatmap
  - Download current view as CSV or PNG
  - Upload your own pre-aggregated CSV to replace the synthetic demo data

Data schema (uploaded CSV):
    Required columns: bin_id, date, position, item, bin_rank, segment,
                      N_item, group_N, pct
    One row per unique [bin_id, date, position, bin_rank, segment, item].
    N_item  = observation count for that item at that key.
    group_N = total observations for [bin_id, date, position, bin_rank, segment]
              (same value for every item row in the same group).
    pct     = N_item / group_N.
    Dates must be in M/D/YYYY format (single- or double-digit month/day).

Aggregation methods
-------------------
  Majority      (M1): per-date plurality winner (most observations on that date),
                      then cross-date majority count; random tiebreak.
  Abs. Majority (M2): per-date: plurality winner keeps its vote only if it holds
                      ≥50 % of that date's observations; otherwise that date votes
                      "VARIOUS". Cross-date: majority count of those per-date votes
                      (VARIOUS is a valid vote value).
  Weighted      (M3): sum(N_item across dates) / sum(group_N across dates) per
                      item; winner = highest aggregate share.
"""

import io
import json
import re
import urllib.parse
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

# ============ CONSTANTS ============
N_MAX_ITEMS      = 12          # total display cap (10 distinct + VARIOUS + _OTHER_)
N_MAX_USER_ITEMS = 10          # distinctly-coloured user items; slots 10=VARIOUS 11=OTHER reserved

VARIOUS_LABEL = 'VARIOUS'      # sentinel string for Abs. Majority fallback
VARIOUS_COLOR = '#C2410C'      # burnt-orange — distinct from all item & other colours
OTHER_LABEL   = '_OTHER_'
OTHER_COLOR   = '#9CA3AF'      # neutral gray

ITEMS = ['APX', 'BRT', 'CFD', 'DLT', 'ETR', 'FRM', 'GVS', 'HXC', 'INV', 'JTL', 'KLP', 'LMR', 'NQT']
COLORS = ['#B91C1C', '#1E3A8A', '#15803D', '#CA8A04', '#6D28D9',
          '#DB2777', '#0E7490', '#525252', '#92400E', '#4D7C0F',
          '#C2410C', '#0369A1']   # index 10 = VARIOUS_COLOR; index 11 = spare

SEGMENTS = ['NA', 'SA', 'EU', 'AS', 'CN', 'AU']
BIN_NAMES = [
    'Apex',  'Basin', 'Birch', 'Bloom', 'Bluff', 'Brace', 'Briar', 'Brook',
    'Brume', 'Cable', 'Cairn', 'Canal', 'Cape',  'Cedar', 'Chalk', 'Cirque',
    'Cleft', 'Cliff', 'Clove', 'Comet', 'Coral', 'Cove',  'Crag',  'Creek',
    'Crest', 'Crown', 'Drift', 'Dune',  'Eddy',  'Elder', 'Elm',   'Ember',
    'Fable', 'Fern',  'Fjord', 'Flint', 'Floe',  'Fold',  'Forge', 'Frost',
    'Glade', 'Glen',  'Gorge', 'Grant', 'Grove', 'Gulf',  'Gulch', 'Heath',
    'Heron', 'Hinge', 'Holt',  'Howe',  'Inlet', 'Isle',  'Ivory', 'Kelp',
    'Knoll', 'Lathe', 'Ledge', 'Loch',  'Lodge', 'Loom',  'Lune',  'Maple',
    'Marsh', 'Mauve', 'Mesa',  'Mill',  'Mire',  'Mist',  'Moat',  'Moor',
    'Morse', 'Moss',  'Nave',  'Notch', 'Opal',  'Orbit', 'Pale',  'Peak',
    'Peat',  'Pine',  'Plane', 'Plume', 'Pond',  'Prism', 'Quill', 'Rapid',
    'Reef',  'Ridge', 'Rime',  'Rune',  'Scarp', 'Sedge', 'Shale', 'Shore',
    'Silt',  'Slate', 'Spire', 'Spur',
]
NUM_BINS      = 100
NUM_POSITIONS = 50
NUM_DATES     = 52    # synthetic dataset uses weekly snapshots

METHOD_OPTIONS = ['Majority', 'Abs. Majority', 'Weighted']

BG    = '#F7F4ED'
INK   = '#1A1A1A'
MUTED = '#6B6B6B'

TITLE_FONTS = {
    "Fraunces":         ("'Fraunces', Georgia, serif",          "italic"),
    "Playfair Display": ("'Playfair Display', Georgia, serif",  "normal"),
    "DM Serif Display": ("'DM Serif Display', Georgia, serif",  "normal"),
    "IBM Plex Sans":    ("'IBM Plex Sans', sans-serif",         "normal"),
}

SORT_GUIDE_URL   = "https://labrou.github.io/bin-selector/sort_modes_explainer.html"
METHOD_GUIDE_URL = "https://labrou.github.io/bin-selector/method_explainer.html"


def sort_descriptions(bt, it):
    return {
        "Index":          f"Alphabetical order of {bt} ID — no analytical grouping; stable baseline.",
        "Similarity":     f"{bt.capitalize()}s sharing the same {it}s at positions 1–4 cluster together. Default.",
        f"{bt.capitalize()} Rank": f"Top = highest-ranked {bt}s (rank 1).",
        "Top-rank":       f"Groups {bt}s sharing the same {it} at position 1; ties resolved by positions 2, 3, …",
        "Selected Share": f"Ranks {bt}s by how many visible positions are held by selected {it}s.",
    }


# ============ DATA GENERATION ============
@st.cache_data
def generate_data():
    """Return synthetic data in compact form — no permanent dense counts cube.

    Primary arrays (for Majority / Abs. Majority):
      date_winner    : (B, D, P) int16   per-date plurality winner; -1 = no data
      date_top_share : (B, D, P) float32 per-date top-item share (max_count/group_N)

    Sparse long arrays (for Weighted; only non-zero item counts stored):
      wt_bin_idx, wt_date_idx, wt_pos_idx : int32
      wt_item_idx                          : int16
      wt_N_item                            : int32

    Each (bin, date, position) cell has k ∈ [1, 10] simulated observations so
    all three aggregation methods can disagree on the same data.
    """
    rng = np.random.default_rng(42)
    n_items = len(ITEMS)

    archetypes = np.array([
        [0.30, 0.16, 0.13, 0.09, 0.08, 0.06, 0.04, 0.03, 0.03, 0.02, 0.03, 0.02, 0.01],
        [0.04, 0.26, 0.20, 0.13, 0.09, 0.07, 0.06, 0.04, 0.02, 0.02, 0.03, 0.02, 0.02],
        [0.07, 0.07, 0.07, 0.19, 0.17, 0.13, 0.09, 0.05, 0.04, 0.04, 0.04, 0.02, 0.02],
        [0.04, 0.04, 0.05, 0.05, 0.07, 0.11, 0.15, 0.17, 0.13, 0.11, 0.04, 0.02, 0.02],
        [0.13, 0.05, 0.15, 0.05, 0.13, 0.07, 0.13, 0.05, 0.09, 0.06, 0.05, 0.02, 0.02],
    ])

    def positional_bias(pos):
        if pos < 5:
            return np.array([3.0, 2.4, 2.0, 1.2, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.6, 0.5, 0.5])
        if pos < 15:
            return np.array([1.2, 1.2, 1.2, 2.0, 2.0, 1.8, 1.0, 1.0, 0.8, 0.8, 0.6, 0.5, 0.5])
        return np.ones(n_items)

    pos_biases = np.array([positional_bias(p) for p in range(NUM_POSITIONS)])

    bin_archetypes   = rng.integers(0, 5, NUM_BINS)
    bin_segments     = np.array(rng.choice(SEGMENTS, NUM_BINS))
    archetype_rank_center = np.array([15, 35, 50, 70, 50])
    rank_noise       = (rng.random(NUM_BINS) + rng.random(NUM_BINS) + rng.random(NUM_BINS) - 1.5) * 18
    _raw_scores      = archetype_rank_center[bin_archetypes] + rank_noise
    bin_ranks        = (np.argsort(np.argsort(_raw_scores)) + 1).astype(int)

    has_regime      = rng.random(NUM_BINS) < 0.20
    regime_dates    = rng.integers(15, 38, NUM_BINS)
    regime_new_arch = rng.integers(0, 5, NUM_BINS)
    drift_direction = rng.normal(0, 1, (NUM_BINS, n_items))
    drift_strength  = 0.30

    end_date = date.today()
    dates    = [end_date - timedelta(weeks=NUM_DATES - 1 - w) for w in range(NUM_DATES)]

    date_idx   = np.arange(NUM_DATES)
    use_regime = has_regime[:, None] & (date_idx[None, :] >= regime_dates[:, None])
    arch_idx   = np.where(use_regime, regime_new_arch[:, None], bin_archetypes[:, None])

    base = archetypes[arch_idx]
    date_factors = date_idx / (NUM_DATES - 1) * drift_strength
    base = base + drift_direction[:, None, :] * date_factors[None, :, None]
    base = np.clip(base, 0.01, None)
    base = base / base.sum(axis=2, keepdims=True)

    combined  = base[:, :, None, :] * pos_biases[None, None, :, :]
    combined /= combined.sum(axis=3, keepdims=True)

    k_obs = rng.integers(1, 11, (NUM_BINS, NUM_DATES, NUM_POSITIONS), dtype=np.int32)

    # Compact output arrays
    date_winner    = np.full((NUM_BINS, NUM_DATES, NUM_POSITIONS), -1, dtype=np.int16)
    date_top_share = np.zeros((NUM_BINS, NUM_DATES, NUM_POSITIONS), dtype=np.float32)

    # Long arrays for Weighted (per-bin lists, concatenated at end)
    wt_bins = []; wt_dates = []; wt_pos = []; wt_items = []; wt_ni = []

    for b in range(NUM_BINS):
        probs_b = combined[b]
        k_b     = k_obs[b]
        max_k   = int(k_b.max())

        cdf_b = np.cumsum(probs_b, axis=-1)
        r     = rng.random((NUM_DATES, NUM_POSITIONS, max_k))
        draws = (r[:, :, :, None] < cdf_b[:, :, None, :]).argmax(axis=-1)

        k_range = np.arange(max_k)
        valid   = k_range[None, None, :] < k_b[:, :, None]

        d_idx, p_idx, _ = np.where(valid)
        item_idx = draws[d_idx, p_idx, np.where(valid)[2]].astype(np.intp)

        # Temporary per-bin counts (discarded after extracting compact arrays)
        counts_b = np.zeros((NUM_DATES, NUM_POSITIONS, n_items), dtype=np.int32)
        np.add.at(counts_b, (d_idx, p_idx, item_idx), 1)

        group_n_b  = counts_b.sum(axis=-1)
        has_data_b = group_n_b > 0
        argmax_b   = counts_b.argmax(axis=-1)      # lower index wins ties
        max_cnt_b  = counts_b.max(axis=-1)

        date_winner[b]    = np.where(has_data_b, argmax_b, -1).astype(np.int16)
        date_top_share[b] = np.where(
            has_data_b,
            (max_cnt_b / np.maximum(group_n_b, 1)).astype(np.float32),
            np.float32(0)
        )

        # Sparse records — keep only non-zero item counts
        di_nz, pi_nz, ii_nz = np.where(counts_b > 0)
        if len(di_nz) > 0:
            wt_bins.append(np.full(len(di_nz), b, dtype=np.int32))
            wt_dates.append(di_nz.astype(np.int32))
            wt_pos.append(pi_nz.astype(np.int32))
            wt_items.append(ii_nz.astype(np.int16))
            wt_ni.append(counts_b[di_nz, pi_nz, ii_nz].astype(np.int32))
        # counts_b is discarded here

    _cat = lambda lst, dt: np.concatenate(lst) if lst else np.array([], dtype=dt)
    return {
        'date_winner':    date_winner,
        'date_top_share': date_top_share,
        'wt_bin_idx':  _cat(wt_bins,  np.int32),
        'wt_date_idx': _cat(wt_dates, np.int32),
        'wt_pos_idx':  _cat(wt_pos,   np.int32),
        'wt_item_idx': _cat(wt_items, np.int16),
        'wt_N_item':   _cat(wt_ni,    np.int32),
        'bin_ranks':    bin_ranks,
        'bin_segments': bin_segments,
        'bin_names':    np.array(BIN_NAMES),
        'dates':        dates,
        'item_codes':   list(ITEMS),
        'item_colors':  list(COLORS[:N_MAX_USER_ITEMS]) + [OTHER_COLOR] * (len(ITEMS) - N_MAX_USER_ITEMS),
    }


@st.cache_data
def discover_items(file_bytes: bytes, filename: str):
    """Fast first pass: return item vocabulary ranked by total N_item."""
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), usecols=['item', 'N_item'])
        df['item']   = df['item'].astype(str)
        df['N_item'] = pd.to_numeric(df['N_item'], errors='coerce').fillna(0)
        vc = df.groupby('item')['N_item'].sum().sort_values(ascending=False)
        return vc.index.tolist(), vc.to_dict()
    except Exception:
        return [], {}


@st.cache_data
def load_user_data(file_bytes: bytes, filename: str):
    """Parse a pre-aggregated CSV into compact date_winner/date_top_share arrays
    and sparse long arrays for the Weighted method.

    Required columns: bin_id, date, position, item, bin_rank, segment,
                      N_item, group_N, pct
    Dates accepted in M/D/YYYY format (single or double-digit month/day).
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        st.error(f"Could not parse CSV: {exc}")
        return None

    required = {'bin_id', 'date', 'position', 'item', 'bin_rank', 'segment', 'N_item', 'group_N'}
    missing  = required - set(df.columns)
    if missing:
        st.error(f"CSV is missing columns: {', '.join(sorted(missing))}")
        return None

    # Date: parse as M/D/YYYY (single- or double-digit month/day, 4-digit year).
    _raw_dates = df['date'].astype(str)
    df['date']  = pd.to_datetime(_raw_dates, format='%m/%d/%Y', errors='coerce')
    _n_nat = df['date'].isna().sum()
    if _n_nat:
        _fallback = pd.to_datetime(_raw_dates[df['date'].isna()], dayfirst=False, errors='coerce')
        df.loc[df['date'].isna(), 'date'] = _fallback
        _still_nat = df['date'].isna().sum()
        if _still_nat:
            st.warning(
                f"{_still_nat:,} row(s) had unparseable dates and will be dropped. "
                f"Expected format: M/D/YYYY (e.g. 1/5/2024 or 12/31/2024)."
            )
            df = df[df['date'].notna()]
    df['date'] = df['date'].dt.date

    df['item']    = df['item'].astype(str)
    df['bin_id']  = df['bin_id'].astype(str)
    df['segment'] = df['segment'].astype(str)
    df['N_item']  = pd.to_numeric(df['N_item'],  errors='coerce').fillna(0).astype(np.int32)
    df['group_N'] = pd.to_numeric(df['group_N'], errors='coerce').fillna(0).astype(np.int32)
    df['bin_key'] = df['bin_id'] + ' · ' + df['segment']

    # Item vocabulary ranked by total observations
    user_items  = (df.groupby('item')['N_item'].sum()
                   .sort_values(ascending=False).index.tolist())
    item_to_idx = {code: i for i, code in enumerate(user_items)}

    bin_keys  = sorted(df['bin_key'].unique())
    dates     = sorted(df['date'].unique())
    positions = sorted(df['position'].unique())

    n_bins  = len(bin_keys)
    n_dates = len(dates)
    n_pos   = len(positions)

    bin_idx_map  = {b: i for i, b in enumerate(bin_keys)}
    date_idx_map = {d: i for i, d in enumerate(dates)}
    pos_idx_map  = {p: i for i, p in enumerate(positions)}

    # Integer index columns (for vectorised array fill below)
    df['_bi'] = df['bin_key'].map(bin_idx_map).astype(np.int32)
    df['_di'] = df['date'].map(date_idx_map).astype(np.int32)
    df['_pi'] = df['position'].map(pos_idx_map).astype(np.int32)
    df['_ii'] = df['item'].map(item_to_idx)          # float, NaN for unknown items

    valid_mask = df['_ii'].notna()
    df_v = df[valid_mask].copy()
    df_v['_ii'] = df_v['_ii'].astype(np.int32)

    # ── date_winner / date_top_share ──────────────────────────────────────────
    # Sort by N_item DESC then item index ASC within each cell so that the
    # plurality winner (tie-break: lower item index) is always the first row.
    df_sorted = df_v.sort_values(
        ['_bi', '_di', '_pi', 'N_item', '_ii'],
        ascending=[True, True, True, False, True],
    )
    winners = df_sorted.drop_duplicates(['_bi', '_di', '_pi'], keep='first')

    date_winner    = np.full((n_bins, n_dates, n_pos), -1, dtype=np.int16)
    date_top_share = np.zeros((n_bins, n_dates, n_pos), dtype=np.float32)

    _bw  = winners['_bi'].to_numpy(np.intp)
    _dw  = winners['_di'].to_numpy(np.intp)
    _pw  = winners['_pi'].to_numpy(np.intp)
    _iw  = winners['_ii'].to_numpy(np.int16)
    _ni_w = winners['N_item'].to_numpy(np.int32)
    _gn_w = winners['group_N'].to_numpy(np.int32)
    _sh   = np.where(_gn_w > 0,
                     _ni_w.astype(np.float32) / np.maximum(_gn_w, 1),
                     np.float32(0))

    date_winner[_bw, _dw, _pw]    = _iw
    date_top_share[_bw, _dw, _pw] = _sh.astype(np.float32)

    # ── Sparse long arrays for Weighted ──────────────────────────────────────
    bin_meta = (
        df.drop_duplicates('bin_key')
        .set_index('bin_key')
        .loc[bin_keys, ['bin_rank', 'segment']]
    )

    return {
        'date_winner':    date_winner,
        'date_top_share': date_top_share,
        'wt_bin_idx':  df_v['_bi'].to_numpy(np.int32),
        'wt_date_idx': df_v['_di'].to_numpy(np.int32),
        'wt_pos_idx':  df_v['_pi'].to_numpy(np.int32),
        'wt_item_idx': df_v['_ii'].to_numpy(np.int16),
        'wt_N_item':   df_v['N_item'].to_numpy(np.int32),
        'bin_ranks':    bin_meta['bin_rank'].to_numpy().astype(int),
        'bin_segments': bin_meta['segment'].to_numpy().astype(str),
        'bin_names':    np.array(bin_keys),
        'dates':        list(dates),
        'item_codes':   user_items,
    }


# ============ COMPUTE FUNCTIONS ============

def compute_plurality(date_winner_slice, n_items):
    """METHOD_1 — Majority.

    Cross-date majority count of per-date plurality winners.
    Tiebreak = most-recent-date winner.

    date_winner_slice : (n_bins, n_dates, n_pos) int16  — -1 = no data
    Returns           : winner (n_bins, n_pos) int16,
                        share  (n_bins, n_pos) float32 = fraction of dates winner won.
    """
    n_bins, n_dates, n_pos = date_winner_slice.shape
    has_data = date_winner_slice >= 0

    if n_dates == 1:
        w = date_winner_slice[:, 0, :].copy()
        s = has_data[:, 0, :].astype(np.float32)
        return w, s

    bd, dd, pd_ = np.where(has_data)
    items_at_valid = date_winner_slice[bd, dd, pd_].astype(np.intp)
    flat_idx   = (bd.astype(np.intp) * n_pos + pd_) * n_items + items_at_valid
    win_counts = np.bincount(flat_idx, minlength=n_bins * n_pos * n_items
                             ).reshape(n_bins, n_pos, n_items).astype(np.int32)

    max_wins = win_counts.max(axis=-1)
    winner   = win_counts.argmax(axis=-1).astype(np.int16)

    recent_has  = has_data[:, -1, :]
    recent_win  = date_winner_slice[:, -1, :]
    b2, p2      = np.meshgrid(np.arange(n_bins), np.arange(n_pos), indexing='ij')
    safe_rw     = np.clip(recent_win, 0, n_items - 1)
    recent_wins = win_counts[b2, p2, safe_rw]
    winner = np.where(recent_has & (recent_wins == max_wins),
                      recent_win, winner).astype(np.int16)

    n_data_dates = has_data.sum(axis=1)
    no_data      = n_data_dates == 0
    winner[no_data] = -1
    share = np.where(no_data, np.float32(0),
                     (max_wins / np.maximum(n_data_dates, 1)).astype(np.float32))
    return winner, share


def compute_abs_majority(date_winner_slice, date_top_share_slice, n_items, various_idx):
    """METHOD_2 — Abs. Majority.

    Per-date: if top-item share >= 0.50 → that item; else → VARIOUS.
    Cross-date: majority count of per-date values (VARIOUS is a valid vote value).
    Tiebreak = most-recent-date value.

    date_winner_slice    : (n_bins, n_dates, n_pos) int16
    date_top_share_slice : (n_bins, n_dates, n_pos) float32
    Returns: winner (n_bins, n_pos) int16,
             share  (n_bins, n_pos) float32 = fraction of dates won by winner.
    """
    n_bins, n_dates, n_pos = date_winner_slice.shape
    has_data = date_winner_slice >= 0

    date_value = np.where(
        ~has_data,                        np.int16(-1),
        np.where(date_top_share_slice >= 0.5,
                 date_winner_slice,       np.int16(various_idx))
    ).astype(np.int16)

    if n_dates == 1:
        return date_value[:, 0, :], date_top_share_slice[:, 0, :].astype(np.float32)

    n_vals = n_items + 1
    bd, dd, pd_ = np.where(has_data)
    vals_valid  = date_value[bd, dd, pd_].astype(np.intp)
    flat_idx    = (bd.astype(np.intp) * n_pos + pd_) * n_vals + vals_valid
    win_counts  = np.bincount(flat_idx, minlength=n_bins * n_pos * n_vals
                              ).reshape(n_bins, n_pos, n_vals).astype(np.int32)

    max_wins = win_counts.max(axis=-1)
    winner   = win_counts.argmax(axis=-1).astype(np.int16)

    recent_has = has_data[:, -1, :]
    recent_val = date_value[:, -1, :]
    b2, p2     = np.meshgrid(np.arange(n_bins), np.arange(n_pos), indexing='ij')
    safe_rv    = np.clip(recent_val, 0, n_vals - 1)
    recent_wins = win_counts[b2, p2, safe_rv]
    winner = np.where(recent_has & (recent_wins == max_wins),
                      recent_val, winner).astype(np.int16)

    n_data_dates = has_data.sum(axis=1)
    no_data      = n_data_dates == 0
    winner[no_data] = -1
    share = np.where(no_data, np.float32(0),
                     (max_wins / np.maximum(n_data_dates, 1)).astype(np.float32))
    return winner, share


def compute_weighted(data, visible_bin_indices, date_start_idx, date_end_idx,
                     pos_indices, n_items):
    """METHOD_3 — Weighted.

    Aggregate the sparse wt_* long arrays for the visible (bins × dates × positions).
    share[b, p, i] = sum_dates(N_item[b, :, p, i]) / sum_dates(group_N[b, :, p])
    Winner = item with highest aggregate share.

    Returns: winner  (n_vis, n_pos_sel) int16,
             share   (n_vis, n_pos_sel) float32  (winner's aggregate share),
             weights (n_vis, n_pos_sel, n_items) float32  (all items' shares — for bar)
    """
    n_vis     = len(visible_bin_indices)
    n_pos_sel = len(pos_indices)
    n_bins_t  = data['date_winner'].shape[0]
    n_pos_t   = data['date_winner'].shape[2]

    _empty = (np.full((n_vis, n_pos_sel), -1, dtype=np.int16),
              np.zeros((n_vis, n_pos_sel), dtype=np.float32),
              np.zeros((n_vis, n_pos_sel, n_items), dtype=np.float32))

    bi = data['wt_bin_idx'];  di = data['wt_date_idx']
    pi = data['wt_pos_idx'];  ii = data['wt_item_idx']
    ni = data['wt_N_item']
    if len(bi) == 0:
        return _empty

    # Lookup tables: global index → local (-1 = not in view)
    vis_local = np.full(n_bins_t, -1, dtype=np.int32)
    vis_local[visible_bin_indices] = np.arange(n_vis, dtype=np.int32)
    pos_local = np.full(n_pos_t, -1, dtype=np.int32)
    pos_local[np.array(pos_indices, dtype=np.int32)] = np.arange(n_pos_sel, dtype=np.int32)

    in_vis  = vis_local[bi] >= 0
    in_date = (di >= date_start_idx) & (di <= date_end_idx)
    in_pos  = pos_local[pi] >= 0
    mask    = in_vis & in_date & in_pos
    if not mask.any():
        return _empty

    lb  = vis_local[bi[mask]]
    lp  = pos_local[pi[mask]]
    iif = ii[mask].astype(np.intp)
    nif = ni[mask]

    flat_idx = (lb.astype(np.intp) * n_pos_sel + lp) * n_items + iif
    total_N  = np.bincount(flat_idx, weights=nif.astype(np.float64),
                           minlength=n_vis * n_pos_sel * n_items
                           ).reshape(n_vis, n_pos_sel, n_items).astype(np.float32)

    group_total = total_N.sum(axis=-1)
    weights     = (total_N / np.maximum(group_total[:, :, None], np.float32(1))).astype(np.float32)
    winner      = weights.argmax(axis=-1).astype(np.int16)
    max_wt      = weights.max(axis=-1).astype(np.float32)

    no_data = group_total == 0
    winner[no_data] = -1
    return winner, np.where(no_data, np.float32(0), max_wt), weights


def compute_view(data, visible_bin_indices, date_start_idx, date_end_idx,
                 pos_indices, method, n_items, various_idx):
    """Dispatch to the appropriate compute function.

    For Majority / Abs. Majority: slices compact date_winner / date_top_share.
    For Weighted: aggregates sparse long arrays over the visible view.

    Returns (winner, share, weights_or_None)
      winner  : (n_vis, n_pos_sel) int16
      share   : (n_vis, n_pos_sel) float32
      weights : (n_vis, n_pos_sel, n_items) float32 | None
    """
    if method == 'Weighted':
        return compute_weighted(data, visible_bin_indices, date_start_idx, date_end_idx,
                                pos_indices, n_items)

    # Compact slice: (n_vis, n_dates_sel, n_pos_sel) — much smaller than 4D counts
    dw = data['date_winner'][visible_bin_indices,
                             date_start_idx:date_end_idx + 1, :][:, :, pos_indices]

    if method == 'Abs. Majority':
        ds = data['date_top_share'][visible_bin_indices,
                                   date_start_idx:date_end_idx + 1, :][:, :, pos_indices]
        w, s = compute_abs_majority(dw, ds, n_items, various_idx)
        return w, s, None

    # Default: Majority
    w, s = compute_plurality(dw, n_items)
    return w, s, None


# ============ HELPERS ============
def dim_color(hex_color, dim_amount=0.88, bg=BG):
    fg     = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    bg_rgb = tuple(int(bg[i:i+2], 16)        for i in (1, 3, 5))
    blended = tuple(int(fg[j] * (1 - dim_amount) + bg_rgb[j] * dim_amount) for j in range(3))
    return f'#{blended[0]:02x}{blended[1]:02x}{blended[2]:02x}'


def apply_url_params(dates, item_codes=None):
    if item_codes is None:
        item_codes = ITEMS
    p = st.query_params
    if not p:
        return

    if 'segments' in p and 'segments_pills' not in st.session_state:
        val = [r for r in p['segments'].split(',') if r]
        if val:
            st.session_state['segments_pills'] = val

    if 'items' in p and 'items_pills' not in st.session_state:
        st.session_state['items_pills'] = [i for i in p['items'].split(',') if i in pill_items]

    if 'ds' in p and 'de' in p and 'wk_slider' not in st.session_state:
        try:
            ds = date.fromisoformat(p['ds'])
            de = date.fromisoformat(p['de'])
            ds_snap = min(dates, key=lambda d: abs((d - ds).days))
            de_snap = min(dates, key=lambda d: abs((d - de).days))
            st.session_state['wk_slider'] = (ds_snap, de_snap)
        except (ValueError, TypeError):
            pass

    if 'rk0' in p and 'rk1' in p and 'rank_slider' not in st.session_state:
        try:
            lo, hi = int(p['rk0']), int(p['rk1'])
            st.session_state['rank_slider'] = (min(lo, hi), max(lo, hi))
        except ValueError:
            pass

    if 'ps0' in p and 'ps1' in p and 'pos_slider' not in st.session_state:
        try:
            lo, hi = int(p['ps0']), int(p['ps1'])
            st.session_state['pos_slider'] = (max(1, min(lo, hi)), max(lo, hi))
        except ValueError:
            pass

    if 'sort' in p and 'sort_radio' not in st.session_state:
        st.session_state['sort_radio'] = p['sort']

    if 'method' in p and 'method_pills' not in st.session_state:
        if p['method'] in METHOD_OPTIONS:
            st.session_state['method_pills'] = p['method']

    if 'cs' in p and 'cell_sz' not in st.session_state:
        try:
            st.session_state['cell_sz'] = max(6, min(28, int(p['cs'])))
        except ValueError:
            pass

    if 'af' in p and 'auto_fit_cb' not in st.session_state:
        st.session_state['auto_fit_cb'] = (p['af'] == '1')


def make_view_csv(bin_names, positions, items_grid, share_grid, ranks, segments,
                  item_codes=None, bin_term='bin', method='Majority',
                  weights_grid=None):
    """Export the current view.  For Weighted, include per-item share columns."""
    if item_codes is None:
        item_codes = ITEMS
    n_bins, n_pos = items_grid.shape
    codes = np.array(item_codes + [VARIOUS_LABEL])
    flat  = items_grid.ravel().astype(int)
    item_labels = np.where(flat >= 0,
                           codes[np.clip(flat, 0, len(codes) - 1)],
                           '')
    share_col = 'weighted_share' if method == 'Weighted' else 'item_share'
    base = pd.DataFrame({
        bin_term:  np.repeat(bin_names, n_pos),
        'rank':    np.repeat(ranks, n_pos).astype(int),
        'segment': np.repeat(segments, n_pos),
        'position': np.tile(positions.astype(int), n_bins),
        'item':     item_labels,
        share_col:  np.round(share_grid.ravel().astype(float), 4),
    })
    if method == 'Weighted' and weights_grid is not None:
        # Build all per-item share columns at once to avoid DataFrame fragmentation
        wt_cols = {
            f'share_{code}': np.round(weights_grid[:, :, i].ravel().astype(float), 4)
            for i, code in enumerate(item_codes)
        }
        base = pd.concat([base, pd.DataFrame(wt_cols, index=base.index)], axis=1)
    return base.to_csv(index=False).encode()


# ============ APP ============
_st_ver = tuple(int(x) for x in st.__version__.split('.')[:2])
_chart_own_width  = {'width': 'content'} if _st_ver >= (1, 51) else {'use_container_width': False}
_chart_full_width = {'width': 'stretch'} if _st_ver >= (1, 51) else {'use_container_width': True}
_btn_full_width   = {'width': 'stretch'} if _st_ver >= (1, 51) else {'use_container_width': True}

st.set_page_config(
    page_title="Ranked Placement Atlas",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_custom_bg      = st.session_state.get('bg_color', BG) or BG
_title_font_key = st.session_state.get('title_font', 'Fraunces')
_title_font_css, _title_font_style = TITLE_FONTS.get(_title_font_key, TITLE_FONTS['Fraunces'])
_custom_title   = st.session_state.get('custom_title', '') or 'Ranked Placement Atlas'

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,400;0,500;1,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&family=Playfair+Display:wght@400;500&family=DM+Serif+Display&display=swap" rel="stylesheet">
<style>
    .stApp {{ background-color: {_custom_bg}; }}
    .main .block-container {{ max-width: 1400px; padding-top: 2rem; }}
    .title-block {{
        border-bottom: 1px solid #2A2A2A;
        padding-bottom: 14px;
        margin-bottom: 18px;
    }}
    .eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: {INK};
        margin-bottom: 4px;
    }}
    .title {{
        font-family: 'Fraunces', Georgia, serif;
        font-weight: 400;
        font-style: italic;
        font-size: 38px;
        line-height: 1.05;
        letter-spacing: -0.02em;
        color: {INK};
        margin: 0;
    }}
    .subtitle {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 13px;
        color: #4A4A4A;
        margin-top: 6px;
        max-width: 700px;
        line-height: 1.5;
    }}
    [data-testid="stWidgetLabel"] p {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        color: {INK} !important;
    }}
    div[data-baseweb="button-group"] button {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.04em !important;
    }}
    .stButton button {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 8px !important;
        letter-spacing: 0.06em !important;
        text-transform: none !important;
        white-space: nowrap !important;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        outline: none !important;
        color: #AAAAAA !important;
        padding: 0px 3px !important;
        min-height: 18px !important;
        line-height: 1.4 !important;
    }}
    .stButton button:hover {{
        color: {INK} !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        text-decoration: underline !important;
    }}
    [data-testid="stHorizontalBlock"] [data-testid="stHorizontalBlock"] {{
        gap: 2px !important;
    }}
    [data-testid="stHorizontalBlock"] [data-testid="stHorizontalBlock"]
        > [data-testid="stColumn"] {{
        padding-left: 2px !important;
        padding-right: 2px !important;
    }}
    div[role="radiogroup"] label {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px !important;
    }}
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] hr {{
        margin-top:    2px !important;
        margin-bottom: 2px !important;
    }}
    [data-testid="stMain"] [data-testid="stMarkdownContainer"]:has(hr) {{
        margin-bottom: 0 !important;
    }}
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {{
        row-gap: 8px !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: {_custom_bg};
    }}
    .stSidebarCollapsedControl,
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {{
        display: flex !important;
        flex-direction: column;
        align-items: center;
    }}
    .stSidebarCollapsedControl::after,
    [data-testid="collapsedControl"]::after,
    [data-testid="stSidebarCollapsedControl"]::after {{
        content: "settings";
        display: block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 8px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {MUTED};
        text-align: center;
        margin-top: 2px;
        pointer-events: none;
    }}
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
data = generate_data()

with st.sidebar:
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:11px;'
        f'color:{MUTED};margin-bottom:4px;">Upload your own data, rename the vocabulary, '
        f'and customise the display — then share a link to your exact view.</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:11px;'
        f'letter-spacing:0.15em;text-transform:uppercase;color:{INK};'
        f'margin-bottom:8px;">Data source</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help=(
            "Required columns: bin_id, date, position, item, bin_rank, segment, "
            "N_item, group_N, pct.  One row per unique "
            "[bin_id, date, position, bin_rank, segment, item] combination."
        ),
        label_visibility="collapsed",
    )
    colored_items = None
    if uploaded is not None:
        items_by_freq, item_counts = discover_items(uploaded.getvalue(), uploaded.name)

        if len(items_by_freq) > N_MAX_USER_ITEMS:
            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:10px;'
                f'color:{MUTED};margin:6px 0 2px;">Your data has {len(items_by_freq)} unique '
                f'item values. Pick up to {N_MAX_USER_ITEMS} to give distinct colours; '
                f'the rest show as gray but still display their real label.</div>',
                unsafe_allow_html=True,
            )
            chosen = st.multiselect(
                "Items to colour distinctly",
                options=items_by_freq,
                default=items_by_freq[:N_MAX_USER_ITEMS],
                format_func=lambda x: f"{x}  ({int(item_counts.get(x, 0)):,})",
                max_selections=N_MAX_USER_ITEMS,
                label_visibility="collapsed",
                key="item_selector",
            )
            colored_items = chosen if chosen else items_by_freq[:N_MAX_USER_ITEMS]
        else:
            colored_items = items_by_freq

        user_data = load_user_data(uploaded.getvalue(), uploaded.name)
        if user_data is not None:
            data = user_data
            n_b, n_d, n_p = data['date_winner'].shape
            n_it = len(data['item_codes'])
            st.success(
                f"{uploaded.name}\n\n"
                f"{n_b} bins · {n_p} positions · {n_d} snapshots · {n_it} items"
            )
    else:
        st.caption("Using synthetic demo data. Upload a pre-aggregated CSV to use your own data.")
        st.caption(
            "Required columns: `bin_id`, `date`, `position`, `item`, `bin_rank`, `segment`, "
            "`N_item`, `group_N`, `pct`  \n"
            "Dates in M/D/YYYY format.  "
            f"Up to {N_MAX_USER_ITEMS} items get distinct colours; extras shown in gray."
        )

    st.divider()
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:11px;'
        f'letter-spacing:0.15em;text-transform:uppercase;color:{INK};'
        f'margin-bottom:8px;">Labels</div>',
        unsafe_allow_html=True,
    )
    bin_term     = st.text_input("Bins are called",    "bin",     key="bin_term").strip()    or "bin"
    item_term    = st.text_input("Items are called",   "item",    key="item_term").strip()   or "item"
    segment_term = st.text_input(f"{bin_term.capitalize()} grouping attribute", "segment",
                                 key="segment_term").strip() or "segment"

    st.divider()
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:11px;'
        f'letter-spacing:0.15em;text-transform:uppercase;color:{INK};'
        f'margin-bottom:8px;">Display</div>',
        unsafe_allow_html=True,
    )
    st.text_input("Title", "Ranked Placement Atlas", key="custom_title")
    st.radio("Title font", list(TITLE_FONTS.keys()), key="title_font")
    st.color_picker("Background color", BG, key="bg_color")

    st.divider()
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:11px;'
        f'letter-spacing:0.15em;text-transform:uppercase;color:{INK};'
        f'margin-bottom:8px;">Share this view</div>',
        unsafe_allow_html=True,
    )
    try:
        _share_host  = st.context.headers.get("host", "localhost:8502")
        _share_proto = "https" if not _share_host.startswith("localhost") else "http"
    except Exception:
        _share_host, _share_proto = "localhost:8502", "http"
    _share_placeholder = st.empty()
    st.caption("Link encodes your current filters, sort mode, method, and date range.")

# ── Item vocabulary ───────────────────────────────────────────────────────────
item_codes = data.get('item_codes', list(ITEMS))
n_items    = len(item_codes)
VARIOUS_IDX = n_items   # sentinel used in winner arrays; beyond all real item indices

if colored_items is not None:
    _colored_set = set(colored_items)
    _color_idx   = {item: i for i, item in enumerate(colored_items)}
    item_colors  = [
        COLORS[_color_idx[item]] if item in _colored_set else OTHER_COLOR
        for item in item_codes
    ]
    pill_items = list(colored_items)
else:
    item_colors = data.get('item_colors', list(COLORS[:n_items]))
    if len(item_colors) < n_items:
        item_colors = item_colors + [OTHER_COLOR] * (n_items - len(item_colors))
    pill_items = [c for c, col in zip(item_codes, item_colors) if col != OTHER_COLOR]
    if not pill_items:
        pill_items = list(item_codes)

_item_sig = ','.join(pill_items)
if st.session_state.get('_item_sig') != _item_sig:
    st.session_state.pop('items_pills', None)
    st.session_state['_item_sig'] = _item_sig

available_segments = sorted(np.unique(data['bin_segments']).tolist())

_segment_sig = ','.join(available_segments)
if st.session_state.get('_segment_sig') != _segment_sig:
    st.session_state.pop('segments_pills', None)
    st.session_state['_segment_sig'] = _segment_sig

# Reset date/rank/pos sliders when the dataset changes so stale values
# from a previous dataset (e.g. synthetic dates) don't crash list.index().
_dataset_sig = (
    f"{data['dates'][0].isoformat()}__{data['dates'][-1].isoformat()}"
    f"__{len(data['dates'])}"
    f"__{int(data['bin_ranks'].min())}__{int(data['bin_ranks'].max())}"
    f"__{data['date_winner'].shape[2]}"   # n_positions
)
if st.session_state.get('_dataset_sig') != _dataset_sig:
    for _k in ('wk_slider', 'rank_slider', 'pos_slider'):
        st.session_state.pop(_k, None)
    st.session_state['_dataset_sig'] = _dataset_sig

_pill_colors   = [item_colors[item_codes.index(it)] for it in pill_items]
_n_gray        = n_items - len(pill_items)
_other_pill    = f'other ({_n_gray})' if _n_gray > 0 else None
_colors_json   = json.dumps(_pill_colors + ([OTHER_COLOR] if _other_pill else []))
_segments_json = json.dumps(available_segments)
_methods_json  = json.dumps(METHOD_OPTIONS)

# JS: colour item pills; skip segment and method pill groups
st.html(
    f"""<script>
(function(){{
  var C={_colors_json}, BG="{_custom_bg}", N=C.length, R={_segments_json}, M={_methods_json}, GRAY="{OTHER_COLOR}";
  function go(){{
    try{{
      var gs=document.querySelectorAll('[data-baseweb="button-group"]');
      for(var i=0;i<gs.length;i++){{
        var bs=gs[i].querySelectorAll('button');
        var realBs=Array.from(bs).filter(function(b){{return !b.getAttribute('data-gray-pill');}});
        if(realBs.length!==N) continue;
        var texts=realBs.map(function(b){{return b.textContent.trim();}});
        // Skip segment and method pill groups
        if(R.some(function(r){{return texts.indexOf(r)>=0;}})) continue;
        if(M.some(function(m){{return texts.indexOf(m)>=0;}})) continue;
        for(var j=0;j<realBs.length;j++){{
          var b=realBs[j];
          var sel=b.getAttribute('aria-pressed')==='true'||b.getAttribute('aria-checked')==='true';
          b.style.setProperty('color', sel ? BG : C[j], 'important');
          b.style.setProperty('border-color', C[j], 'important');
          b.style.setProperty('background-color', sel ? C[j] : '', 'important');
        }}
      }}
    }}catch(e){{}}
  }}
  go();
  try{{
    new MutationObserver(go).observe(document.body,
      {{subtree:true,childList:true,attributes:true,attributeFilter:['aria-pressed','aria-checked']}});
  }}catch(e){{ setInterval(go,200); }}
}})();
</script>"""
)

apply_url_params(data['dates'], item_codes)

n_pos_total  = data['date_winner'].shape[2]
min_rank_val = int(data['bin_ranks'].min())
max_rank_val = int(data['bin_ranks'].max())


# ── Dialogs ───────────────────────────────────────────────────────────────────
@st.dialog("User Guide", width="large")
def _show_user_guide():
    st.markdown(f"""
### What you're looking at

A heatmap where each **row** is a {bin_term}, each **column** is a ranked position,
and each **cell** is coloured by the {item_term} occupying that slot.
When you select a date range spanning multiple snapshots, each cell is resolved
according to the active **Method** (see the Method section in Row 1).
Hover any cell for exact values.

Below the heatmap is a stacked bar showing the {item_term} distribution
across visible {bin_term}s at each position (interpretation varies by method).

---

### Method — Row 1

| Method | What it does |
|---|---|
| **Majority** | For each date, the plurality winner (most observations); across dates, the item that won most dates is shown. |
| **Abs. Majority** | Per date: the plurality winner wins only if it holds ≥ 50 % of that day's observations; otherwise that day's value is **VARIOUS**. Cross-date aggregation is identical to Majority. |
| **Weighted** | Pools all observations across the date range; item share = total N\_item ÷ total group\_N. Winner = highest share. |

[Visual guide to all methods →]({METHOD_GUIDE_URL})

---

### Filters — Row 1

**{segment_term.capitalize()}s** · Toggleable pills. Selecting a subset hides {bin_term}s whose {segment_term} is not selected.

**{item_term.capitalize()}s** · Toggleable pills. Controls *highlighting*, not filtering —
all {item_term}s remain visible but unselected ones are dimmed.

---

### Ranges — Row 2

**Date** · Dual-handle slider. A single date shows that snapshot exactly;
a wider range triggers the selected Method's aggregation logic.

**{bin_term.capitalize()} rank range** · Filter rows by global rank.

**Position range** · Hide columns outside the chosen window.

---

### Sort — Row 3

| Mode | What it does |
|---|---|
| **Index** | Original data order |
| **Similarity** | Groups {bin_term}s that share the same {item_term}s at positions 1–4 (default) |
| **{bin_term.capitalize()} Rank** | Ascending by global rank |
| **Top-rank** | Groups {bin_term}s sharing the same {item_term} at position 1; ties resolved by position 2, 3, … |
| **Selected Share** | Ranks {bin_term}s by how many visible positions are held by selected {item_term}s |

[Visual guide to all sort modes →]({SORT_GUIDE_URL})

---

### Drill-down

Click any cell to select that {bin_term}. A time-series heatmap appears below,
showing how its {item_term} mix evolved across every date in the selected range.
Hover shows the {item_term}'s share for that specific date.

---

### Uploading your own data

Open the **sidebar** and upload a pre-aggregated CSV with these columns:

| Column | Notes |
|---|---|
| `bin_id` | Display name |
| `date` | M/D/YYYY format (single or double-digit month/day) |
| `position` | Ranked position within the {bin_term} |
| `item` | Any string label |
| `bin_rank` | Global rank of the {bin_term} |
| `segment` | Grouping / filter attribute |
| `N_item` | Observation count for this item at this key |
| `group_N` | Total observations for [bin\_id, date, position, bin\_rank, segment] |
| `pct` | N\_item / group\_N |

One row per unique [bin\_id, date, position, bin\_rank, segment, **item**].
""")


@st.dialog("Sort modes — visual guide", width="large")
def _show_sort_guide():
    try:
        html = Path(__file__).with_name("sort_modes_explainer.html").read_text(encoding="utf-8")
        style_m = re.search(r'<style>(.*?)</style>', html, re.DOTALL)
        body_m  = re.search(r'<body>(.*?)</body>',   html, re.DOTALL)
        if style_m and body_m:
            st.html(f"<style>{style_m.group(1)}</style>{body_m.group(1)}")
        else:
            st.html(html)
    except FileNotFoundError:
        st.markdown(f"[Open visual guide in browser →]({SORT_GUIDE_URL})")


@st.dialog("Method — visual guide", width="large")
def _show_method_guide():
    try:
        html = Path(__file__).with_name("method_explainer.html").read_text(encoding="utf-8")
        style_m = re.search(r'<style>(.*?)</style>', html, re.DOTALL)
        body_m  = re.search(r'<body>(.*?)</body>',   html, re.DOTALL)
        if style_m and body_m:
            st.html(f"<style>{style_m.group(1)}</style>{body_m.group(1)}")
        else:
            st.html(html)
    except FileNotFoundError:
        st.markdown(f"[Open method guide in browser →]({METHOD_GUIDE_URL})")


# ── Title ─────────────────────────────────────────────────────────────────────
_title_col, _help_col = st.columns([9, 1])
with _title_col:
    n_bins_total  = data['date_winner'].shape[0]
    n_dates_total = data['date_winner'].shape[1]
    st.markdown(f"""
<div class="title-block">
    <div class="title" style="font-family:{_title_font_css};font-style:{_title_font_style};">{_custom_title}</div>
    <div class="subtitle">{n_bins_total} {bin_term}s &times; {n_pos_total} positions &times;
    {n_items} {item_term}s &times; {n_dates_total} snapshots.</div>
</div>
""", unsafe_allow_html=True)
with _help_col:
    st.write("")
    st.write("")
    if st.button("User guide", key="help_btn"):
        _show_user_guide()

# ── Row 1: Segments · Items · Method ──────────────────────────────────────────
col_segments, col_items, col_method = st.columns(3)

with col_segments:
    if 'segments_pills' not in st.session_state:
        st.session_state['segments_pills'] = list(available_segments)
    else:
        stored = st.session_state['segments_pills']
        valid  = [r for r in stored if r in available_segments]
        if stored and not valid:
            st.session_state['segments_pills'] = list(available_segments)
        elif valid != stored:
            st.session_state['segments_pills'] = valid

    _all_reg  = list(available_segments)
    def _reg_all():  st.session_state['segments_pills'] = _all_reg
    def _reg_none(): st.session_state['segments_pills'] = []

    _shdr, _sall, _snone, _ = st.columns([4, 1, 1, 8], gap="small")
    with _shdr:
        st.markdown(
            f'<p style="font-family:IBM Plex Mono,monospace;font-size:11px;'
            f'letter-spacing:0.15em;text-transform:uppercase;color:{INK};'
            f'margin:0;padding-top:5px;">{segment_term.capitalize()}s</p>',
            unsafe_allow_html=True,
        )
    with _sall:  st.button("all",  key="btn_reg_all",  on_click=_reg_all)
    with _snone: st.button("none", key="btn_reg_none", on_click=_reg_none)
    selected_segments = st.pills(
        f"{segment_term.capitalize()}s",
        available_segments,
        selection_mode="multi",
        key="segments_pills",
        label_visibility="collapsed",
    )

with col_items:
    if 'items_pills' not in st.session_state:
        st.session_state['items_pills'] = list(pill_items)

    _all_items = list(pill_items)
    def _items_all():  st.session_state['items_pills'] = _all_items
    def _items_none(): st.session_state['items_pills'] = []
    def _strip_other():
        if _other_pill and _other_pill in (st.session_state.get('items_pills') or []):
            st.session_state['items_pills'] = [
                s for s in st.session_state['items_pills'] if s != _other_pill
            ]

    _ihdr, _iall, _inone, _ = st.columns([4, 1, 1, 8], gap="small")
    with _ihdr:
        st.markdown(
            f'<p style="font-family:IBM Plex Mono,monospace;font-size:11px;'
            f'letter-spacing:0.15em;text-transform:uppercase;color:{INK};'
            f'margin:0;padding-top:5px;">{item_term.capitalize()}s</p>',
            unsafe_allow_html=True,
        )
    with _iall:  st.button("all",  key="btn_all",   on_click=_items_all)
    with _inone: st.button("none", key="btn_clear", on_click=_items_none)

    _pill_display = pill_items + ([_other_pill] if _other_pill else [])
    selected_items = st.pills(
        f"{item_term.capitalize()}s",
        _pill_display,
        selection_mode="multi",
        key="items_pills",
        on_change=_strip_other,
        label_visibility="collapsed",
    )
    if _other_pill:
        selected_items = [s for s in (selected_items or []) if s != _other_pill]

with col_method:
    # Initialise default
    if 'method_pills' not in st.session_state:
        st.session_state['method_pills'] = 'Majority'
    elif st.session_state['method_pills'] not in METHOD_OPTIONS:
        st.session_state['method_pills'] = 'Majority'

    _mhdr, _mq, _ = st.columns([4, 2, 8], gap="small")
    with _mhdr:
        st.markdown(
            f'<p style="font-family:IBM Plex Mono,monospace;font-size:11px;'
            f'letter-spacing:0.15em;text-transform:uppercase;color:{INK};'
            f'margin:0;padding-top:5px;">Method</p>',
            unsafe_allow_html=True,
        )
    with _mq:
        if st.button("method guide", key="method_guide_btn"):
            _show_method_guide()

    method = st.pills(
        "Method",
        METHOD_OPTIONS,
        selection_mode="single",
        key="method_pills",
        label_visibility="collapsed",
    ) or 'Majority'

# ── Row 2: Ranges — Date / Bin Rank / Position ────────────────────────────────
st.divider()

col_date, col_rank, col_pos = st.columns(3)

_n  = len(data['dates'])
_wk = st.session_state.pop('wk_slider', None)
if not isinstance(_wk, (list, tuple)):
    _wk = (data['dates'][max(0, _n - 13)], data['dates'][-1])

with col_date:
    date_range = st.select_slider(
        "Date",
        options=data['dates'],
        value=_wk,
        format_func=lambda d: d.strftime("%b %d, '%y"),
        key="wk_slider",
    )

_rk = st.session_state.pop('rank_slider', None)
if not isinstance(_rk, (list, tuple)) or _rk[0] < min_rank_val or _rk[1] > max_rank_val or _rk[0] > _rk[1]:
    _rk = (min_rank_val, max_rank_val)

with col_rank:
    rank_range = st.slider(
        f"{bin_term.capitalize()} rank range", min_rank_val, max_rank_val,
        value=_rk, key="rank_slider",
    )

_ps = st.session_state.pop('pos_slider', None)
if not isinstance(_ps, (list, tuple)) or _ps[1] > n_pos_total:
    _ps = (1, n_pos_total)

with col_pos:
    pos_range = st.slider(
        "Position range", 1, n_pos_total,
        value=_ps, key="pos_slider",
    )

st.divider()

# ── Row 3: Sort ───────────────────────────────────────────────────────────────
n_sel        = len(selected_items) if selected_items else 0
n_pill_items = len(pill_items)

sort_options = ["Index", "Similarity", f"{bin_term.capitalize()} Rank", "Top-rank"]
if 0 < n_sel < n_pill_items:
    sort_options.append("Selected Share")

# Pop-before-render: avoids default-vs-session-state conflict on st.radio.
_sort = st.session_state.pop('sort_radio', None)
if _sort not in sort_options:
    _sort = 'Similarity'

_sort_lbl_col, _sort_q_col, _ = st.columns([1.4, 1.0, 7.6], gap="small")
with _sort_lbl_col:
    st.markdown(
        f'<p style="font-family:IBM Plex Mono,monospace;font-size:11px;'
        f'letter-spacing:0.15em;text-transform:uppercase;color:{INK};'
        f'margin:0;padding-top:2px;">Sort {bin_term}s by</p>',
        unsafe_allow_html=True,
    )
with _sort_q_col:
    if st.button("sort guide", key="sort_guide_btn"):
        _show_sort_guide()

sort_mode = st.radio(
    f"Sort {bin_term}s by",
    sort_options,
    index=sort_options.index(_sort),
    horizontal=True,
    key="sort_radio",
    label_visibility="collapsed",
)
st.markdown(
    f'<p style="font-family:IBM Plex Sans,sans-serif;font-size:13px;'
    f'color:#666666;margin:10px 0 14px 0;line-height:1.5;">'
    f'{sort_descriptions(bin_term, item_term).get(sort_mode, "")}</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── Filtering ─────────────────────────────────────────────────────────────────
segments_active     = selected_segments if selected_segments else available_segments
in_segment          = np.isin(data['bin_segments'], segments_active)
in_rank             = (data['bin_ranks'] >= rank_range[0]) & (data['bin_ranks'] <= rank_range[1])
visible_mask        = in_segment & in_rank
visible_bin_indices = np.where(visible_mask)[0]

date_start_idx = data['dates'].index(date_range[0])
date_end_idx   = data['dates'].index(date_range[1])
date_indices   = list(range(date_start_idx, date_end_idx + 1))

pos_indices = list(range(pos_range[0] - 1, pos_range[1]))

if len(visible_bin_indices) == 0 or len(date_indices) == 0 or len(pos_indices) == 0:
    st.warning("No data in current filter range. Widen your selectors.")
    st.stop()

# ── Compute view (session-state cache) ─────────────────────────────────────
# M1/M2: slices compact (B,D,P) arrays — sub-millisecond even without cache.
# M3: aggregates sparse long arrays; cache avoids re-filtering on sort/highlight.
_view_sig = (
    st.session_state.get('_dataset_sig', ''),
    tuple(visible_bin_indices.tolist()),
    date_start_idx, date_end_idx,
    tuple(pos_indices),
    method, n_items,
)
if st.session_state.get('_view_sig') != _view_sig:
    majority_view, share_view, weights_view = compute_view(
        data, visible_bin_indices, date_start_idx, date_end_idx,
        pos_indices, method, n_items, VARIOUS_IDX,
    )
    st.session_state.update({
        '_view_sig':  _view_sig,
        '_maj_view':  majority_view,
        '_shr_view':  share_view,
        '_wgt_view':  weights_view,
    })
else:
    majority_view = st.session_state['_maj_view']
    share_view    = st.session_state['_shr_view']
    weights_view  = st.session_state['_wgt_view']

# ── Sort ──────────────────────────────────────────────────────────────────────
n_vis    = len(visible_bin_indices)
sort_arr = np.where(majority_view == VARIOUS_IDX, -1, majority_view).astype(np.int32)

if sort_mode == "Index":
    order = np.arange(n_vis)
elif sort_mode == f"{bin_term.capitalize()} Rank":
    order = np.argsort(data['bin_ranks'][visible_bin_indices], kind='stable')
elif sort_mode == "Similarity":
    top4  = sort_arr[:, :4]
    order = np.lexsort(top4.T[::-1])
elif sort_mode == "Top-rank":
    order = np.lexsort(sort_arr.T[::-1])
elif sort_mode == "Selected Share":
    sel_idx_set  = [item_codes.index(i) for i in selected_items]
    sel_mask     = np.isin(majority_view, sel_idx_set)
    share_count  = sel_mask.sum(axis=1)
    pos_sum      = (sel_mask * (np.array(pos_indices) + 1)).sum(axis=1)
    order        = np.lexsort([pos_sum, -share_count])
else:
    order = np.arange(n_vis)

ordered_bin_indices = visible_bin_indices[order]
majority_disp       = majority_view[order]
share_disp          = share_view[order]
weights_disp        = weights_view[order] if weights_view is not None else None

n_show_bins = len(ordered_bin_indices)
n_show_pos  = len(pos_indices)

# ── Colorscale ────────────────────────────────────────────────────────────────
sel_idx_set = set(item_codes.index(i) for i in selected_items) if selected_items else set()
all_or_none = (len(sel_idx_set) == 0) or (len(sel_idx_set) >= n_pill_items)

def effective_color(i):
    if item_colors[i] == OTHER_COLOR:
        return OTHER_COLOR
    if all_or_none or i in sel_idx_set:
        return item_colors[i]
    return dim_color(item_colors[i], 0.88, bg=_custom_bg)

_eff_colors = [effective_color(i) for i in range(n_items)]
# VARIOUS always at full color (not dimmed)
_eff_colors_with_various = _eff_colors + [VARIOUS_COLOR]

# Colorscale: slots [-1, 0..n_items-1, n_items(VARIOUS)]
# zmin=-1, zmax=n_items+1  →  (n_items+2) equal-width bands
_N = n_items + 2
colorscale = [[0.0, _custom_bg], [1 / _N, _custom_bg]]   # band 0: no-data
for i in range(n_items):
    colorscale.append([(i + 1) / _N, _eff_colors[i]])
    colorscale.append([(i + 2) / _N, _eff_colors[i]])
# VARIOUS band
colorscale.append([(n_items + 1) / _N, VARIOUS_COLOR])
colorscale.append([1.0,               VARIOUS_COLOR])

# ── Sizing ────────────────────────────────────────────────────────────────────
cell_size = st.session_state.get('cell_sz', 12)
auto_fit  = st.session_state.get('auto_fit_cb', True)

container_w, container_h = 900, 720
if auto_fit:
    cell_w = max(cell_size, container_w / max(n_show_pos, 1))
    cell_h = max(cell_size, container_h / max(n_show_bins, 1))
else:
    cell_w = cell_h = cell_size

cell_w = min(cell_w, 40)
cell_h = min(cell_h, 30)
heatmap_width  = int(n_show_pos  * cell_w)
heatmap_height = int(n_show_bins * cell_h)
total_width    = heatmap_width + 170
total_height   = int(heatmap_height / 0.83) + 60

# ── Display arrays ────────────────────────────────────────────────────────────
positions_disp = np.array(pos_indices) + 1
ranks_disp     = data['bin_ranks'][ordered_bin_indices]
segments_disp  = data['bin_segments'][ordered_bin_indices]
bin_names_disp = data['bin_names'][ordered_bin_indices]

_codes_with_various = np.array(item_codes + [VARIOUS_LABEL])
text_grid = np.where(
    majority_disp == VARIOUS_IDX,
    VARIOUS_LABEL,
    np.where(
        majority_disp >= 0,
        _codes_with_various[np.clip(majority_disp.astype(int), 0, n_items)],
        '—'
    )
)
z = majority_disp.astype(float)

customdata = np.empty((n_show_bins, n_show_pos, 4), dtype=object)
customdata[:, :, 0] = ranks_disp[:, None].astype(int)
customdata[:, :, 1] = segments_disp[:, None]
customdata[:, :, 2] = share_disp.astype(float)
# slot 3: per-cell note shown in hover; only non-blank for M2 VARIOUS cells
if method == 'Abs. Majority':
    customdata[:, :, 3] = np.where(
        majority_disp == VARIOUS_IDX,
        ' (no per-date majority)',
        '',
    )
else:
    customdata[:, :, 3] = ''

# ── View summary ──────────────────────────────────────────────────────────────
date_count = len(date_indices)
multi_date = date_count > 1

d0 = date_range[0].strftime("%b %d, %Y")
if multi_date:
    d1 = date_range[1].strftime("%b %d, %Y")
    method_desc = {
        'Majority':      f"Each cell = {item_term} that won the most individual snapshots (per-date plurality).",
        'Abs. Majority': f"Each cell = per-date plurality winner if it holds ≥ 50 % of that day's observations; otherwise <b>VARIOUS</b>. Cross-date: majority count of those per-date values.",
        'Weighted':      f"Each cell = {item_term} with the highest aggregate share across all observations in the window.",
    }[method]
    mode_sentence = (
        f"{method_desc}  "
        f"Range: {d0} → {d1} ({date_count} snapshots)."
    )
    _date_label = f"{date_count} snapshots"
else:
    mode_sentence = f"Single snapshot ({d0})."
    _date_label = "1 snapshot"

share_label = {
    'Weighted':      'Weighted share',   # aggregate N_item / group_N across dates
    'Majority':      'Date-win share',   # fraction of dates won by plurality winner
    'Abs. Majority': 'Date-win share',   # fraction of dates that voted for this value (incl. VARIOUS)
}[method]

summary_html = f"""
<div style="background:rgba(0,0,0,0.04);border-radius:4px;
            padding:12px 16px;margin-bottom:8px;max-width:900px;">
    <p style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;
              font-weight:600;color:{INK};margin:0 0 6px 0;">Showing
        · Method: <b>{method}</b></p>
    <div style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;color:{INK};line-height:1.6;">
        <b>{n_show_bins}</b> {bin_term}{'s' if n_show_bins != 1 else ''} ·
        <b>{n_show_pos}</b> position{'s' if n_show_pos != 1 else ''} ·
        {_date_label} ·
        {segment_term}s: <b>{', '.join(sorted(set(segments_active)))}</b> ·
        sort: <b>{sort_mode}</b>
        <div style="margin-top:5px;color:#666666;">{mode_sentence}</div>
    </div>
</div>
"""
st.markdown(summary_html, unsafe_allow_html=True)

# ── Legend ────────────────────────────────────────────────────────────────────
_item_code_idx = {code: i for i, code in enumerate(item_codes)}
legend_parts   = []
for item in pill_items:
    idx  = _item_code_idx[item]
    c    = _eff_colors[idx]
    dim  = not (all_or_none or idx in sel_idx_set)
    text_color = MUTED if dim else INK
    legend_parts.append(
        f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px;">'
        f'<span style="display:inline-block;width:11px;height:11px;background:{c};'
        f'flex-shrink:0;border-radius:2px;"></span>'
        f'<span style="color:{text_color};font-family:IBM Plex Mono,monospace;'
        f'font-size:11px;">{item}</span>'
        f'</span>'
    )
n_gray = n_items - len(pill_items)
if n_gray > 0:
    legend_parts.append(
        f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px;">'
        f'<span style="display:inline-block;width:11px;height:11px;background:{OTHER_COLOR};'
        f'flex-shrink:0;border-radius:2px;"></span>'
        f'<span style="color:{MUTED};font-family:IBM Plex Mono,monospace;'
        f'font-size:11px;">{n_gray} other</span>'
        f'</span>'
    )
# VARIOUS legend entry (only relevant for Abs. Majority)
if method == 'Abs. Majority':
    legend_parts.append(
        f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px;">'
        f'<span style="display:inline-block;width:11px;height:11px;background:{VARIOUS_COLOR};'
        f'flex-shrink:0;border-radius:2px;"></span>'
        f'<span style="color:{INK};font-family:IBM Plex Mono,monospace;'
        f'font-size:11px;">VARIOUS</span>'
        f'</span>'
    )
st.markdown(
    '<div style="display:flex;flex-wrap:wrap;gap:2px;margin-top:10px;margin-bottom:4px;">'
    + ''.join(legend_parts) + '</div>',
    unsafe_allow_html=True,
)

# CSV bytes — cached; rebuild only when the displayed view changes
_csv_sig = (
    st.session_state.get('_dataset_sig', ''),
    tuple(ordered_bin_indices.tolist()),
    date_start_idx, date_end_idx,
    tuple(pos_indices), method,
)
if st.session_state.get('_csv_sig') != _csv_sig:
    _new_csv = make_view_csv(
        bin_names_disp, positions_disp, majority_disp, share_disp,
        ranks_disp, segments_disp,
        item_codes=item_codes, bin_term=bin_term,
        method=method, weights_grid=weights_disp,
    )
    st.session_state['_csv_sig']      = _csv_sig
    st.session_state['_csv_bytes_dl'] = _new_csv
csv_bytes = st.session_state.get('_csv_bytes_dl', b'')

# ── Build figure ──────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.85, 0.15],
    vertical_spacing=0.025,
    shared_xaxes=True,
)

fig.add_trace(
    go.Heatmap(
        z=z,
        x=list(positions_disp),
        y=list(bin_names_disp),
        colorscale=colorscale,
        zmin=-1,
        zmax=n_items + 1,
        showscale=False,
        customdata=customdata,
        text=text_grid,
        hovertemplate=(
            "<b>%{y}</b>  ·  Rank %{customdata[0]}  ·  %{customdata[1]}<br>"
            f"Position %{{x}}  ·  {item_term.capitalize()} <b>%{{text}}</b>%{{customdata[3]}}<br>"
            f"{share_label}: %{{customdata[2]:.0%}}"
            "<extra></extra>"
        ),
        xgap=0.5,
        ygap=0.5,
    ),
    row=1, col=1,
)

# ── Bar chart ─────────────────────────────────────────────────────────────────
_colored_idx = [i for i in range(n_items) if item_colors[i] != OTHER_COLOR]
_other_idx   = [i for i in range(n_items) if item_colors[i] == OTHER_COLOR]
_n_bins_safe = max(n_show_bins, 1)

if method == 'Weighted' and weights_disp is not None:
    # Average weighted share per item per position across visible bins
    bar_y_label = f'avg. share · {item_term}'
    for item_idx in _colored_idx:
        fig.add_trace(
            go.Bar(
                x=positions_disp,
                y=weights_disp[:, :, item_idx].mean(axis=0),
                orientation='v',
                marker=dict(color=_eff_colors[item_idx], line=dict(width=0)),
                name=item_codes[item_idx],
                showlegend=False,
                hovertemplate=(
                    f"<b>{item_codes[item_idx]}</b>  avg share: %{{y:.1%}} · pos %{{x}}"
                    "<extra></extra>"
                ),
            ),
            row=2, col=1,
        )
    if _other_idx:
        other_shares = weights_disp[:, :, _other_idx].sum(axis=-1).mean(axis=0)
        fig.add_trace(
            go.Bar(
                x=positions_disp,
                y=other_shares,
                orientation='v',
                marker=dict(color=OTHER_COLOR, line=dict(width=0)),
                name='Other',
                showlegend=False,
                hovertemplate="<b>Other</b>  avg share: %{y:.1%} · pos %{x}<extra></extra>",
            ),
            row=2, col=1,
        )
else:
    # M1 / M2: fraction of bins where item is the winner
    bar_y_label = f'% of {bin_term}s · winner'
    for item_idx in _colored_idx:
        fig.add_trace(
            go.Bar(
                x=positions_disp,
                y=(majority_disp == item_idx).sum(axis=0) / _n_bins_safe,
                orientation='v',
                marker=dict(color=_eff_colors[item_idx], line=dict(width=0)),
                name=item_codes[item_idx],
                showlegend=False,
                hovertemplate=(
                    f"<b>{item_codes[item_idx]}</b> is winner in %{{y:.0%}} of {bin_term}s · pos %{{x}}"
                    "<extra></extra>"
                ),
            ),
            row=2, col=1,
        )
    if _other_idx:
        _other_count = np.isin(majority_disp, _other_idx).sum(axis=0) / _n_bins_safe
        fig.add_trace(
            go.Bar(
                x=positions_disp,
                y=_other_count,
                orientation='v',
                marker=dict(color=OTHER_COLOR, line=dict(width=0)),
                name='Other',
                showlegend=False,
                hovertemplate=f"<b>Other</b> is winner in %{{y:.0%}} of {bin_term}s · pos %{{x}}<extra></extra>",
            ),
            row=2, col=1,
        )
    # VARIOUS bar for Abs. Majority
    if method == 'Abs. Majority':
        various_frac = (majority_disp == VARIOUS_IDX).sum(axis=0) / _n_bins_safe
        if various_frac.max() > 0:
            fig.add_trace(
                go.Bar(
                    x=positions_disp,
                    y=various_frac,
                    orientation='v',
                    marker=dict(color=VARIOUS_COLOR, line=dict(width=0)),
                    name=VARIOUS_LABEL,
                    showlegend=False,
                    hovertemplate=f"<b>VARIOUS</b> in %{{y:.0%}} of {bin_term}s · pos %{{x}}<extra></extra>",
                ),
                row=2, col=1,
            )

xtickvals = sorted(set(
    [int(positions_disp[0]), int(positions_disp[-1])]
    + [p for p in positions_disp if int(p) % 5 == 0]
))

fig.update_layout(
    barmode='stack',
    width=total_width,
    height=total_height,
    margin=dict(l=130, r=20, t=10, b=40),
    plot_bgcolor=_custom_bg,
    paper_bgcolor=_custom_bg,
    showlegend=False,
    bargap=0.08,
    font=dict(family='IBM Plex Sans', size=11, color=INK),
    hoverlabel=dict(
        bgcolor=INK, bordercolor=INK,
        font=dict(family='IBM Plex Mono', size=11, color=BG),
    ),
)
fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
fig.update_xaxes(
    showgrid=False, zeroline=False,
    tickvals=xtickvals, ticktext=[str(v) for v in xtickvals],
    tickfont=dict(size=9, family='IBM Plex Mono', color=MUTED),
    title=dict(text='POSITION', font=dict(size=9, family='IBM Plex Mono', color=MUTED)),
    row=2, col=1,
)
fig.update_yaxes(
    showgrid=False, zeroline=False,
    autorange='reversed',
    tickfont=dict(size=9, family='IBM Plex Mono', color=INK),
    row=1, col=1,
)
fig.update_yaxes(
    showgrid=False, zeroline=False, range=[0, 1],
    tickvals=[0, 0.5, 1], ticktext=['0%', '50%', '100%'],
    tickfont=dict(size=9, family='IBM Plex Mono', color=MUTED),
    title=dict(text=bar_y_label, font=dict(size=9, family='IBM Plex Mono', color=MUTED)),
    row=2, col=1,
)

# ── Render ─────────────────────────────────────────────────────────────────────
chart_event = st.plotly_chart(
    fig, **_chart_own_width, on_select="rerun", key="main_chart",
)

# ── HTML export ────────────────────────────────────────────────────────────────
_html_view_key = (
    tuple(ordered_bin_indices.tolist()),
    date_start_idx, date_end_idx,
    tuple(pos_indices),
    sort_mode,
    method,
    tuple(sorted(sel_idx_set)),
    _custom_bg,
)
if st.session_state.get('_html_view_key') != _html_view_key:
    st.session_state['_html_view_key'] = _html_view_key
    st.session_state.pop('_html_bytes', None)

# ── Chart footer ───────────────────────────────────────────────────────────────
_sz_col, _af_col, _gap_col, _dl_csv_col, _dl_html_col = st.columns([3, 1, 1, 1, 1])
with _sz_col:
    st.slider("Cell size (px)", 6, 28, cell_size, key="cell_sz")
with _af_col:
    st.write("")
    st.checkbox("Auto-fit", value=auto_fit, key="auto_fit_cb")
with _dl_csv_col:
    st.download_button(
        "Download CSV", csv_bytes,
        file_name="atlas_view.csv", mime="text/csv",
        **_btn_full_width,
    )
with _dl_html_col:
    if '_html_bytes' in st.session_state:
        st.download_button(
            "Download heatmap",
            st.session_state['_html_bytes'],
            file_name="atlas_view.html", mime="text/html",
            **_btn_full_width,
        )
    elif st.button("Export heatmap", **_btn_full_width):
        st.session_state['_html_bytes'] = fig.to_html(
            include_plotlyjs='cdn', config={'displayModeBar': True}
        ).encode()
        st.rerun()

# ── Drill-down ────────────────────────────────────────────────────────────────
clicked_bin_name = None
if chart_event and chart_event.selection and chart_event.selection.points:
    for pt in chart_event.selection.points:
        if pt.get('curve_number', -1) == 0 and pt.get('y'):
            clicked_bin_name = pt['y']
            break

drill_col, _ = st.columns([2, 3])
with drill_col:
    _no_sel      = f"— select a {bin_term} —"
    drill_options = [_no_sel] + list(bin_names_disp)
    default_idx   = 0
    if clicked_bin_name and clicked_bin_name in drill_options:
        default_idx = drill_options.index(clicked_bin_name)
    drill_bin = st.selectbox(
        f"Drill down — {bin_term} time series",
        drill_options,
        index=default_idx,
        key="drill_select",
    )

if drill_bin != _no_sel:
    bin_matches = np.where(data['bin_names'] == drill_bin)[0]
    if len(bin_matches) > 0:
        bidx       = bin_matches[0]
        bin_rank_v = data['bin_ranks'][bidx]
        bin_seg    = data['bin_segments'][bidx]

        drill_dates = data['dates'][date_start_idx:date_end_idx + 1]

        # Per-date winner and top-item share from compact arrays (no 4D counts needed)
        drill_winner = data['date_winner'][bidx,
                           date_start_idx:date_end_idx + 1, :][:, pos_indices].copy()
        drill_share  = data['date_top_share'][bidx,
                           date_start_idx:date_end_idx + 1, :][:, pos_indices].copy()
        # shape: (n_dates_sel, n_pos_sel)

        # For Abs. Majority: mark VARIOUS where top-item share < 50 %
        if method == 'Abs. Majority':
            drill_winner = np.where(
                (drill_winner >= 0) & (drill_share < 0.5),
                np.int16(VARIOUS_IDX),
                drill_winner,
            )

        drill_text = np.where(
            drill_winner == VARIOUS_IDX,
            VARIOUS_LABEL,
            np.where(
                drill_winner >= 0,
                _codes_with_various[np.clip(drill_winner.astype(int), 0, n_items)],
                '—'
            )
        )

        _segment_suffix = f" · {bin_seg}" if ' · ' not in drill_bin else ""
        with st.expander(
            f"Time series · {drill_bin} · Rank {bin_rank_v}{_segment_suffix}",
            expanded=True,
        ):
            mini_z    = drill_winner.T.astype(float)       # (n_pos, n_dates_sel)
            mini_text = drill_text.T

            # For M2 the stored value is always the per-date plurality item's pct
            # (which is < 50 % for VARIOUS cells), so label it accordingly.
            if method == 'Weighted':
                drill_share_label = "Pct"
            elif method == 'Abs. Majority':
                drill_share_label = "Plurality item pct"
            else:
                drill_share_label = "Share"
            # Build share customdata for hover
            mini_share = drill_share.T   # (n_pos, n_dates_sel)

            x_labels = [d.strftime("%b %d") for d in drill_dates]
            y_labels = list(positions_disp)

            mini_fig = go.Figure(go.Heatmap(
                z=mini_z,
                x=x_labels,
                y=y_labels,
                colorscale=colorscale,
                zmin=-1, zmax=n_items + 1,
                showscale=False,
                text=mini_text,
                customdata=mini_share[:, :, None],
                hovertemplate=(
                    "Date: <b>%{x}</b>  ·  Position: <b>%{y}</b><br>"
                    f"{item_term.capitalize()}: <b>%{{text}}</b><br>"
                    f"{drill_share_label}: %{{customdata[0]:.0%}}"
                    "<extra></extra>"
                ),
                xgap=0.5, ygap=0.5,
            ))

            mini_cell_w = max(10, min(30, 700 // max(len(date_indices), 1)))
            mini_cell_h = max(6,  min(20, 400 // max(n_show_pos, 1)))
            mini_w = int(len(date_indices) * mini_cell_w) + 80
            mini_h = int(n_show_pos       * mini_cell_h) + 60

            mini_fig.update_layout(
                width=mini_w, height=mini_h,
                margin=dict(l=50, r=20, t=10, b=60),
                plot_bgcolor=_custom_bg, paper_bgcolor=_custom_bg,
                font=dict(family='IBM Plex Sans', size=11, color=INK),
                hoverlabel=dict(
                    bgcolor=INK, bordercolor=INK,
                    font=dict(family='IBM Plex Mono', size=11, color=BG),
                ),
            )
            mini_fig.update_yaxes(
                autorange='reversed', showgrid=False, zeroline=False,
                tickfont=dict(size=9, family='IBM Plex Mono', color=INK),
                title=dict(text='POSITION', font=dict(size=9, family='IBM Plex Mono', color=MUTED)),
            )
            mini_fig.update_xaxes(
                showgrid=False, zeroline=False, tickangle=45,
                tickfont=dict(size=8, family='IBM Plex Mono', color=MUTED),
            )
            st.plotly_chart(mini_fig, **_chart_own_width)

            # CSV for this bin's time series
            n_di, n_dp = drill_winner.shape
            drill_csv = pd.DataFrame({
                bin_term:   drill_bin,
                'rank':     int(bin_rank_v),
                'segment':  bin_seg,
                'date':     np.repeat([d.isoformat() for d in drill_dates], n_dp),
                'position': np.tile(positions_disp.astype(int), n_di),
                'item':     drill_text.ravel(),
                'share':    np.round(drill_share.ravel(), 4),
            }).to_csv(index=False).encode()
            st.download_button(
                f"Download {drill_bin} time series CSV",
                drill_csv,
                file_name=f"{drill_bin.lower()}_timeseries.csv",
                mime="text/csv",
            )

# ── URL query params ───────────────────────────────────────────────────────────
_new_params = {
    'segments': ','.join(sorted(set(segments_active))),
    'items':    ','.join(selected_items) if selected_items else '',
    'ds':       date_range[0].isoformat(),
    'de':       date_range[1].isoformat(),
    'rk0':      str(rank_range[0]),
    'rk1':      str(rank_range[1]),
    'ps0':      str(pos_range[0]),
    'ps1':      str(pos_range[1]),
    'sort':     sort_mode,
    'method':   method,
    'cs':       str(cell_size),
    'af':       '1' if auto_fit else '0',
}
if dict(st.query_params) != _new_params:
    st.query_params.update(_new_params)

# ── Share button ───────────────────────────────────────────────────────────────
_qs_share  = urllib.parse.urlencode(_new_params, doseq=True)
_share_url = f"{_share_proto}://{_share_host}/" + (f"?{_qs_share}" if _qs_share else "")
with _share_placeholder.container():
    st.code(_share_url, language=None)
