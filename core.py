"""
core.py — Streamlit-free computation logic for Ranked Placement Atlas.

Import from here to keep app.py thin and to allow unit-testing without
a running Streamlit server.
"""

import hashlib
import io
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd


# ============ CSV PARSING ============
def parse_uploaded_csv(file_bytes: bytes) -> tuple:
    """Parse a pre-aggregated CSV into compact arrays for Ranked Placement Atlas.

    Required columns: bin_id, date, position, item, bin_rank, segment
    Optional columns: N_item (defaults to 1 per row if absent)
                      filter (row-level provenance label)

    Returns (data_dict, messages) where messages is a list of (level, text) tuples.
    level is one of "error", "warning", "success".
    Returns (None, messages) on fatal error.
    """
    messages = []

    try:
        df = pd.read_csv(io.BytesIO(file_bytes), keep_default_na=False, na_values=[''])
    except Exception as exc:
        messages.append(("error", f"Could not parse CSV: {exc}"))
        return None, messages

    required = {'bin_id', 'date', 'position', 'item', 'bin_rank', 'segment'}
    missing  = required - set(df.columns)
    if missing:
        messages.append(("error", f"CSV is missing columns: {', '.join(sorted(missing))}"))
        return None, messages

    has_filter = 'filter' in df.columns

    # Date: parse as M/D/YYYY (single- or double-digit month/day, 4-digit year).
    _raw_dates = df['date'].astype(str)
    df['date']  = pd.to_datetime(_raw_dates, format='%m/%d/%Y', errors='coerce')
    _n_nat = df['date'].isna().sum()
    if _n_nat:
        _fallback = pd.to_datetime(_raw_dates[df['date'].isna()], dayfirst=False, errors='coerce')
        df.loc[df['date'].isna(), 'date'] = _fallback
        _still_nat = df['date'].isna().sum()
        if _still_nat:
            messages.append((
                "warning",
                f"{_still_nat:,} row(s) had unparseable dates and will be dropped. "
                f"Expected format: M/D/YYYY (e.g. 1/5/2024 or 12/31/2024)."
            ))
            df = df[df['date'].notna()]
    df['date'] = df['date'].dt.date

    # Drop rows where required string fields are null before astype(str)
    for _col in ('item', 'bin_id', 'segment'):
        _null = df[_col].isna().sum()
        if _null:
            messages.append(("warning", f"{_null:,} row(s) had missing '{_col}' values and will be dropped."))
            df = df[df[_col].notna()]
    df['item']     = df['item'].astype(str)
    df['bin_id']   = df['bin_id'].astype(str)
    df['segment']  = df['segment'].astype(str)
    if has_filter:
        _null_f = df['filter'].isna().sum()
        if _null_f:
            messages.append(("warning", f"{_null_f:,} row(s) had missing 'filter' values and will be dropped."))
            df = df[df['filter'].notna()]
        df['filter'] = df['filter'].astype(str)
        filter_vals = sorted(df['filter'].unique().tolist())
        n_filters   = len(filter_vals)
        _fi_map     = {fv: i for i, fv in enumerate(filter_vals)}
        df['_fi']   = df['filter'].map(_fi_map).astype(np.int32)
    else:
        filter_vals = []
        n_filters   = 0
    df['bin_rank'] = pd.to_numeric(df['bin_rank'], errors='coerce')
    _bad_rank = df['bin_rank'].isna().sum()
    if _bad_rank:
        messages.append(("warning", f"{_bad_rank:,} row(s) had unparseable bin_rank values; they will be set to 0."))
        df['bin_rank'] = df['bin_rank'].fillna(0)
    _frac_rank = (df['bin_rank'].notna() & (df['bin_rank'] != df['bin_rank'].round())).sum()
    if _frac_rank:
        messages.append((
            "warning",
            f"{_frac_rank:,} row(s) had fractional bin_rank values; "
            "they will be rounded to the nearest integer (e.g. 2.7 → 3)."
        ))
        df['bin_rank'] = df['bin_rank'].round()
    df['bin_rank'] = df['bin_rank'].astype(int)
    df['position'] = pd.to_numeric(df['position'], errors='coerce')
    _bad_pos = df['position'].isna().sum()
    if _bad_pos:
        messages.append(("warning", f"{_bad_pos:,} row(s) had unparseable position values and will be dropped."))
        df = df[df['position'].notna()]
    _frac_pos = (df['position'] != df['position'].round()).sum()
    if _frac_pos:
        messages.append((
            "warning",
            f"{_frac_pos:,} row(s) had fractional position values and will be dropped "
            "(e.g. 1.9 is not a valid integer position; truncation could merge distinct positions)."
        ))
        df = df[df['position'] == df['position'].round()]
    df['position'] = df['position'].astype(int)
    # N_item is optional; default to 1 per row when absent or unparseable
    if 'N_item' in df.columns:
        df['N_item'] = pd.to_numeric(df['N_item'], errors='coerce').fillna(1)
        _frac_ni = (df['N_item'].notna() & (df['N_item'] != df['N_item'].round())).sum()
        if _frac_ni:
            messages.append((
                "warning",
                f"{_frac_ni:,} row(s) had fractional N_item values; "
                "they will be rounded to the nearest integer (e.g. 3.7 → 4)."
            ))
            df['N_item'] = df['N_item'].round()
        df['N_item'] = df['N_item'].astype(np.int32)
        _neg = (df['N_item'] < 0).sum()
        if _neg:
            messages.append(("warning", f"{_neg:,} row(s) had negative N_item values and were set to 0."))
            df['N_item'] = df['N_item'].clip(lower=0)
        # Drop zero-count rows
        _zero = (df['N_item'] == 0).sum()
        if _zero:
            df = df[df['N_item'] > 0]
    else:
        df['N_item'] = np.int32(1)

    # Guard: nothing left after cleaning
    if df.empty:
        messages.append(("error", "No valid rows remain after cleaning. Check dates, positions, and N_item values."))
        return None, messages

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

    df['_bi'] = df['bin_key'].map(bin_idx_map).astype(np.int32)
    df['_di'] = df['date'].map(date_idx_map).astype(np.int32)
    df['_pi'] = df['position'].map(pos_idx_map).astype(np.int32)
    df['_ii'] = df['item'].map(item_to_idx)          # float, NaN for unknown items

    valid_mask = df['_ii'].notna()
    df_v = df[valid_mask].copy()
    df_v['_ii'] = df_v['_ii'].astype(np.int32)

    _agg_key = ['_bi', '_di', '_pi', '_ii'] + (['_fi'] if has_filter else [])
    df_v = (
        df_v.groupby(_agg_key, as_index=False)['N_item'].sum()
    )

    _gn_key = ['_bi', '_di', '_pi'] + (['_fi'] if has_filter else [])
    df_v['group_N'] = df_v.groupby(_gn_key)['N_item'].transform('sum')

    _sort_cols = ['_bi', '_di', '_pi', 'N_item', '_ii']
    _sort_asc  = [True, True, True, False, True]

    def _fill_winner_arrays(df_sub):
        dw  = np.full((n_bins, n_dates, n_pos), -1, dtype=np.int32)
        dts = np.zeros((n_bins, n_dates, n_pos), dtype=np.float32)
        if len(df_sub) == 0:
            return dw, dts
        _s  = df_sub.sort_values(_sort_cols, ascending=_sort_asc)
        _w  = _s.drop_duplicates(['_bi', '_di', '_pi'], keep='first')
        _bw   = _w['_bi'].to_numpy(np.intp)
        _dw_  = _w['_di'].to_numpy(np.intp)
        _pw   = _w['_pi'].to_numpy(np.intp)
        _iw   = _w['_ii'].to_numpy(np.int32)
        _ni_w = _w['N_item'].to_numpy(np.int32)
        _gn_w = _w['group_N'].to_numpy(np.int32)
        _sh   = np.where(_gn_w > 0,
                         _ni_w.astype(np.float32) / np.maximum(_gn_w, 1),
                         np.float32(0))
        dw[_bw, _dw_, _pw]  = _iw
        dts[_bw, _dw_, _pw] = _sh.astype(np.float32)
        _zg = _gn_w <= 0
        if _zg.any():
            dw[_bw[_zg], _dw_[_zg], _pw[_zg]] = -1
        return dw, dts

    if has_filter:
        _dw_list, _dts_list = [], []
        for _fi_i in range(n_filters):
            _dw_i, _dts_i = _fill_winner_arrays(df_v[df_v['_fi'] == _fi_i])
            _dw_list.append(_dw_i)
            _dts_list.append(_dts_i)
        date_winner_by_filter    = np.stack(_dw_list)
        date_top_share_by_filter = np.stack(_dts_list)
        date_winner = date_top_share = None
    else:
        date_winner, date_top_share = _fill_winner_arrays(df_v)
        date_winner_by_filter = date_top_share_by_filter = None

    # Sparse long arrays for Weighted
    bin_meta = (
        df.drop_duplicates('bin_key')
        .set_index('bin_key')
        .loc[bin_keys, ['bin_rank', 'segment']]
    )

    _rank_nunique = df.groupby('bin_key')['bin_rank'].nunique()
    _bad_ranks = (_rank_nunique > 1).sum()
    if _bad_ranks:
        messages.append((
            "warning",
            f"{_bad_ranks} bin(s) have inconsistent bin_rank values across rows; "
            "the first value encountered is used."
        ))

    _file_id = hashlib.md5(file_bytes).hexdigest()[:10]

    data_dict = {
        'date_winner':              date_winner,
        'date_top_share':           date_top_share,
        'date_winner_by_filter':    date_winner_by_filter,
        'date_top_share_by_filter': date_top_share_by_filter,
        'wt_bin_idx':    df_v['_bi'].to_numpy(np.int32),
        'wt_date_idx':   df_v['_di'].to_numpy(np.int32),
        'wt_pos_idx':    df_v['_pi'].to_numpy(np.int32),
        'wt_item_idx':   df_v['_ii'].to_numpy(np.int32),
        'wt_N_item':     df_v['N_item'].to_numpy(np.int32),
        'wt_filter_idx': df_v['_fi'].to_numpy(np.int32) if has_filter else None,
        'bin_ranks':    bin_meta['bin_rank'].to_numpy().astype(int),
        'bin_segments': bin_meta['segment'].to_numpy().astype(str),
        'bin_names':    np.array(bin_keys),
        'dates':        list(dates),
        'positions':    list(positions),
        'item_codes':   user_items,
        'filter_values': filter_vals,
        '_id':          _file_id,
    }
    return data_dict, messages


# ============ SORT ORDER ============
def compute_sort_order(sort_mode, majority_view, bin_ranks, n_items,
                       selected_items, item_codes, pos_indices, n_pill_items):
    """Compute the sort order for the bin rows.

    Parameters
    ----------
    sort_mode       : str — one of "Index", "Similarity", "<X> Rank", "Top-rank",
                      "Selected Share"
    majority_view   : (n_vis, n_pos) int32 — aggregated winner array
    bin_ranks       : (n_vis,) int array — ranks for the visible bins
    n_items         : int — VARIOUS sentinel is >= n_items
    selected_items  : list[str] — selected item codes
    item_codes      : list[str] — full item vocabulary
    pos_indices     : list[int] — active position indices
    n_pill_items    : int — total number of distinctly-coloured items

    Returns
    -------
    order : (n_vis,) int array of row indices in sorted order
    """
    n_vis = majority_view.shape[0]
    # Replace VARIOUS (== n_items) with -1 for sort comparisons
    sort_arr = np.where(majority_view == n_items, -1, majority_view).astype(np.int32)

    if sort_mode == "Index":
        order = np.arange(n_vis)
    elif sort_mode.endswith(" Rank") and sort_mode != "Top-rank":
        # "<X> Rank" modes — use bin_ranks ascending
        order = np.argsort(bin_ranks, kind='stable')
    elif sort_mode == "Similarity":
        top4  = sort_arr[:, :4]
        order = np.lexsort(top4.T[::-1])
    elif sort_mode == "Top-rank":
        order = np.lexsort(sort_arr.T[::-1])
    elif sort_mode == "Selected Share":
        sel_idx_set  = [item_codes.index(i) for i in selected_items if i in item_codes]
        sel_mask     = np.isin(majority_view, sel_idx_set)
        share_count  = sel_mask.sum(axis=1)
        pos_sum      = (sel_mask * (np.array(pos_indices) + 1)).sum(axis=1)
        order        = np.lexsort([pos_sum, -share_count])
    else:
        order = np.arange(n_vis)

    return order

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

SORT_GUIDE_URL   = "static/sort_modes_explainer.html"
METHOD_GUIDE_URL = "static/method_explainer.html"
VIZ_GUIDE_URL    = "static/visualization_explainer.html"


def sort_descriptions(bt, it):
    return {
        "Index":          f"Alphabetical order of {bt} ID — no analytical grouping; stable baseline.",
        "Similarity":     f"{bt.capitalize()}s sharing the same {it}s at positions 1–4 cluster together. Default.",
        f"{bt.capitalize()} Rank": f"Top = highest-ranked {bt}s (rank 1).",
        "Top-rank":       f"Groups {bt}s sharing the same {it} at position 1; ties resolved by positions 2, 3, …",
        "Selected Share": f"Ranks {bt}s by how many visible positions are held by selected {it}s.",
    }


# ============ DATA GENERATION ============
def generate_data():
    """Return synthetic data in compact form — no permanent dense counts cube.

    Primary arrays (for Majority / Abs. Majority):
      date_winner    : (B, D, P) int32   per-date plurality winner; -1 = no data
      date_top_share : (B, D, P) float32 per-date top-item share (max_count/group_N)

    Sparse long arrays (for Weighted; only non-zero item counts stored):
      wt_bin_idx, wt_date_idx, wt_pos_idx : int32
      wt_item_idx                          : int32
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
    date_winner    = np.full((NUM_BINS, NUM_DATES, NUM_POSITIONS), -1, dtype=np.int32)
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

        date_winner[b]    = np.where(has_data_b, argmax_b, -1).astype(np.int32)
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
            wt_items.append(ii_nz.astype(np.int32))
            wt_ni.append(counts_b[di_nz, pi_nz, ii_nz].astype(np.int32))
        # counts_b is discarded here

    _cat = lambda lst, dt: np.concatenate(lst) if lst else np.array([], dtype=dt)
    return {
        'date_winner':    date_winner,
        'date_top_share': date_top_share,
        'wt_bin_idx':  _cat(wt_bins,  np.int32),
        'wt_date_idx': _cat(wt_dates, np.int32),
        'wt_pos_idx':  _cat(wt_pos,   np.int32),
        'wt_item_idx': _cat(wt_items, np.int32),
        'wt_N_item':   _cat(wt_ni,    np.int32),
        'bin_ranks':    bin_ranks,
        'bin_segments': bin_segments,
        'bin_names':    np.array(BIN_NAMES),
        'dates':        dates,
        'positions':    list(range(1, NUM_POSITIONS + 1)),
        'item_codes':   list(ITEMS),
        'item_colors':  list(COLORS[:N_MAX_USER_ITEMS]) + [OTHER_COLOR] * (len(ITEMS) - N_MAX_USER_ITEMS),
        '_id':          'synthetic',
    }


# ============ COMPUTE FUNCTIONS ============

def compute_plurality(date_winner_slice, n_items):
    """METHOD_1 — Majority.

    Cross-date majority count of per-date plurality winners.
    Tiebreak = most-recent-date winner.

    date_winner_slice : (n_bins, n_dates, n_pos) int32  — -1 = no data
    Returns           : winner (n_bins, n_pos) int32,
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
    winner   = win_counts.argmax(axis=-1).astype(np.int32)

    recent_has  = has_data[:, -1, :]
    recent_win  = date_winner_slice[:, -1, :]
    b2, p2      = np.meshgrid(np.arange(n_bins), np.arange(n_pos), indexing='ij')
    safe_rw     = np.clip(recent_win, 0, n_items - 1)
    recent_wins = win_counts[b2, p2, safe_rw]
    winner = np.where(recent_has & (recent_wins == max_wins),
                      recent_win, winner).astype(np.int32)

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

    date_winner_slice    : (n_bins, n_dates, n_pos) int32
    date_top_share_slice : (n_bins, n_dates, n_pos) float32
    Returns: winner (n_bins, n_pos) int32,
             share  (n_bins, n_pos) float32 = fraction of dates won by winner.
    """
    n_bins, n_dates, n_pos = date_winner_slice.shape
    has_data = date_winner_slice >= 0

    date_value = np.where(
        ~has_data,                        np.int32(-1),
        np.where(date_top_share_slice >= 0.5,
                 date_winner_slice,       np.int32(various_idx))
    ).astype(np.int32)

    if n_dates == 1:
        return date_value[:, 0, :], date_top_share_slice[:, 0, :].astype(np.float32)

    n_vals = n_items + 1
    bd, dd, pd_ = np.where(has_data)
    vals_valid  = date_value[bd, dd, pd_].astype(np.intp)
    flat_idx    = (bd.astype(np.intp) * n_pos + pd_) * n_vals + vals_valid
    win_counts  = np.bincount(flat_idx, minlength=n_bins * n_pos * n_vals
                              ).reshape(n_bins, n_pos, n_vals).astype(np.int32)

    max_wins = win_counts.max(axis=-1)
    winner   = win_counts.argmax(axis=-1).astype(np.int32)

    recent_has = has_data[:, -1, :]
    recent_val = date_value[:, -1, :]
    b2, p2     = np.meshgrid(np.arange(n_bins), np.arange(n_pos), indexing='ij')
    safe_rv    = np.clip(recent_val, 0, n_vals - 1)
    recent_wins = win_counts[b2, p2, safe_rv]
    winner = np.where(recent_has & (recent_wins == max_wins),
                      recent_val, winner).astype(np.int32)

    n_data_dates = has_data.sum(axis=1)
    no_data      = n_data_dates == 0
    winner[no_data] = -1
    share = np.where(no_data, np.float32(0),
                     (max_wins / np.maximum(n_data_dates, 1)).astype(np.float32))
    return winner, share


def compute_weighted(data, visible_bin_indices, date_start_idx, date_end_idx,
                     pos_indices, n_items, filter_idx=None, date_winner_arr=None):
    """METHOD_3 — Weighted.

    Aggregate the sparse wt_* long arrays for the visible (bins × dates × positions).
    share[b, p, i] = sum_dates(N_item[b, :, p, i]) / sum_dates(group_N[b, :, p])
    Winner = item with highest aggregate share.

    filter_idx     : int | None — when set, only sparse rows with wt_filter_idx == filter_idx
                     are used (i.e. the selected provenance).
    date_winner_arr: (n_bins, n_dates, n_pos) int32 | None — filter-specific dense array
                     used for tiebreaking; falls back to data['date_winner'] when None.

    Returns: winner  (n_vis, n_pos_sel) int32,
             share   (n_vis, n_pos_sel) float32  (winner's aggregate share),
             weights (n_vis, n_pos_sel, n_items) float32  (all items' shares — for bar)
    """
    n_vis     = len(visible_bin_indices)
    n_pos_sel = len(pos_indices)
    _dw_ref   = date_winner_arr if date_winner_arr is not None else data['date_winner']
    n_bins_t  = _dw_ref.shape[0]
    n_pos_t   = _dw_ref.shape[2]

    _empty = (np.full((n_vis, n_pos_sel), -1, dtype=np.int32),
              np.zeros((n_vis, n_pos_sel), dtype=np.float32),
              np.zeros((n_vis, n_pos_sel, n_items), dtype=np.float32))

    bi = data['wt_bin_idx'];  di = data['wt_date_idx']
    pi = data['wt_pos_idx'];  ii = data['wt_item_idx']
    ni = data['wt_N_item']
    if filter_idx is not None and data.get('wt_filter_idx') is not None:
        _fmask = data['wt_filter_idx'] == filter_idx
        bi, di, pi, ii, ni = bi[_fmask], di[_fmask], pi[_fmask], ii[_fmask], ni[_fmask]
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
    winner      = weights.argmax(axis=-1).astype(np.int32)
    max_wt      = weights.max(axis=-1).astype(np.float32)

    # Tiebreak: when two items share the same max weight, prefer the most recent
    # date's winner — consistent with M1 and M2 behaviour.
    recent_dw = _dw_ref[visible_bin_indices, date_end_idx, :][:, pos_indices]
    # recent_dw shape: (n_vis, n_pos_sel)
    safe_recent = np.clip(recent_dw, 0, n_items - 1)
    b_range, p_range = np.indices((n_vis, n_pos_sel))
    # A cell is a "tie" if the recent winner has the same weight as the argmax winner
    recent_tied = (
        (recent_dw >= 0)                                        # recent date has data
        & (weights[b_range, p_range, safe_recent] == max_wt)   # its weight equals max
    )
    winner = np.where(recent_tied, recent_dw, winner).astype(np.int32)

    no_data = group_total == 0
    winner[no_data] = -1
    return winner, np.where(no_data, np.float32(0), max_wt), weights


def compute_view(data, visible_bin_indices, date_start_idx, date_end_idx,
                 pos_indices, method, n_items, various_idx,
                 filter_idx=None, date_winner_arr=None, date_top_share_arr=None):
    """Dispatch to the appropriate compute function.

    For Majority / Abs. Majority: slices compact date_winner / date_top_share.
    For Weighted: aggregates sparse long arrays over the visible view.

    filter_idx        : int | None  — active filter index (None = no filter column).
    date_winner_arr   : (n_bins, n_dates, n_pos) int32 | None  — filter-specific array;
                        falls back to data['date_winner'] when None.
    date_top_share_arr: same shape, float32 | None.

    Returns (winner, share, weights_or_None)
      winner  : (n_vis, n_pos_sel) int32
      share   : (n_vis, n_pos_sel) float32
      weights : (n_vis, n_pos_sel, n_items) float32 | None
    """
    _dw  = date_winner_arr    if date_winner_arr    is not None else data['date_winner']
    _dts = date_top_share_arr if date_top_share_arr is not None else data['date_top_share']

    if method == 'Weighted':
        return compute_weighted(data, visible_bin_indices, date_start_idx, date_end_idx,
                                pos_indices, n_items,
                                filter_idx=filter_idx, date_winner_arr=_dw)

    # Compact slice: (n_vis, n_dates_sel, n_pos_sel) — much smaller than 4D counts
    dw = _dw[visible_bin_indices,
             date_start_idx:date_end_idx + 1, :][:, :, pos_indices]

    if method == 'Abs. Majority':
        ds = _dts[visible_bin_indices,
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


def make_view_csv(bin_names, positions, items_grid, share_grid, ranks, segments,
                  item_codes=None, bin_term='bin', method='Majority',
                  weights_grid=None, colored_item_codes=None):
    """Export the current view.

    For Weighted, include one share_<item> column per colored item only
    (capped to the distinctly-colored vocabulary to avoid CSV width explosion
    on uploads with hundreds/thousands of unique items).
    """
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
        # Only export share columns for the distinctly-colored items; exporting
        # one column per item in a large vocabulary would produce an unmanageably
        # wide CSV and consume significant memory.
        export_items = colored_item_codes if colored_item_codes is not None else item_codes
        item_to_idx  = {code: i for i, code in enumerate(item_codes)}
        wt_cols = {
            f'share_{code}': np.round(
                weights_grid[:, :, item_to_idx[code]].ravel().astype(float), 4
            )
            for code in export_items
            if code in item_to_idx
        }
        if wt_cols:
            base = pd.concat([base, pd.DataFrame(wt_cols, index=base.index)], axis=1)
    return base.to_csv(index=False).encode()
