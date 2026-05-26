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
  - Item highlight (dims non-selected items)
  - Legend mapping item codes to colors
  - Click any bin row to open its full time-series heatmap
  - Download current view as CSV or PNG
  - Upload your own CSV to replace the synthetic demo data

Data schema (synthetic):
    100 bins × 52 snapshots × 50 ranked positions × 10 items (3-char codes).
    Each bin carries:
        - bin_rank: global rank 1-100 (ties allowed)
        - segment:  one of NA, SA, EU, AS, CN, AU (constant per bin)
    Drift model:
        - Slow archetype drift across weeks
        - 20% of bins undergo a one-time regime change between weeks 15-37
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
N_MAX_ITEMS  = 12          # total display cap including _OTHER_
N_MAX_USER_ITEMS = N_MAX_ITEMS - 1   # user-selectable items; last slot reserved for _OTHER_
OTHER_LABEL  = '_OTHER_'   # synthetic label for items outside the kept vocabulary
OTHER_COLOR  = '#9CA3AF'   # neutral gray — visually distinct from all real item colors

ITEMS = ['APX', 'BRT', 'CFD', 'DLT', 'ETR', 'FRM', 'GVS', 'HXC', 'INV', 'JTL', 'KLP', 'LMR', 'NQT']
COLORS = ['#B91C1C', '#1E3A8A', '#15803D', '#CA8A04', '#6D28D9',
          '#DB2777', '#0E7490', '#525252', '#92400E', '#4D7C0F',
          '#C2410C', '#0369A1']  # two extra slots for uploaded data
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
NUM_WEEKS     = 52

BG       = '#F7F4ED'
INK      = '#1A1A1A'
MUTED    = '#6B6B6B'

TITLE_FONTS = {
    "Fraunces":         ("'Fraunces', Georgia, serif",          "italic"),
    "Playfair Display": ("'Playfair Display', Georgia, serif",  "normal"),
    "DM Serif Display": ("'DM Serif Display', Georgia, serif",  "normal"),
    "IBM Plex Sans":    ("'IBM Plex Sans', sans-serif",         "normal"),
}

SORT_GUIDE_URL = "https://labrou.github.io/bin-selector/sort_modes_explainer.html"

def sort_descriptions(bt, it):
    """Return sort-mode description strings using the active bin/item terminology."""
    return {
        "Index":          f"Alphabetical order of {bt} ID — no analytical grouping; stable baseline.",
        "Similarity":     f"{bt.capitalize()}s sharing the same {it}s at positions 1–4 cluster together, surfacing archetypes as broad horizontal color bands. Default.",
        f"{bt.capitalize()} Rank": f"Top = highest-ranked {bt}s (rank 1). Use this when the question is about rank: do top {bt}s share a distinct profile?",
        "Top-rank":       f"Groups {bt}s that share the same {it} at position 1; ties are resolved by position 2, then 3, and so on — a strict left-to-right sort. Use this to find {bt}s with an identical opening sequence.",
        "Selected Share": f"Ranks {bt}s by how many visible positions are held by the selected {it}s; ties broken by which {bt} has them at earlier positions. Available when 1 to N−1 {it}s are highlighted.",
    }

# ============ DATA GENERATION ============
@st.cache_data
def generate_data():
    rng = np.random.default_rng(42)

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
        return np.ones(len(ITEMS))

    pos_biases = np.array([positional_bias(p) for p in range(NUM_POSITIONS)])  # (50, 10)

    bin_archetypes  = rng.integers(0, 5, NUM_BINS)
    bin_segments     = np.array(rng.choice(SEGMENTS, NUM_BINS))
    archetype_rank_center = np.array([15, 35, 50, 70, 50])
    rank_noise      = (rng.random(NUM_BINS) + rng.random(NUM_BINS) + rng.random(NUM_BINS) - 1.5) * 18
    _raw_scores     = archetype_rank_center[bin_archetypes] + rank_noise
    bin_ranks       = (np.argsort(np.argsort(_raw_scores)) + 1).astype(int)  # dense 1..NUM_BINS

    has_regime      = rng.random(NUM_BINS) < 0.20
    regime_weeks    = rng.integers(15, 38, NUM_BINS)
    regime_new_arch = rng.integers(0, 5, NUM_BINS)
    drift_direction = rng.normal(0, 1, (NUM_BINS, len(ITEMS)))
    drift_strength  = 0.30

    end_date = date.today()
    dates    = [end_date - timedelta(weeks=NUM_WEEKS - 1 - w) for w in range(NUM_WEEKS)]

    # Vectorized: compute archetype index for every (bin, week) simultaneously
    week_idx   = np.arange(NUM_WEEKS)
    use_regime = has_regime[:, None] & (week_idx[None, :] >= regime_weeks[:, None])  # (B, W)
    arch_idx   = np.where(use_regime, regime_new_arch[:, None], bin_archetypes[:, None])  # (B, W)

    base = archetypes[arch_idx]  # (B, W, 10)
    week_factors = week_idx / (NUM_WEEKS - 1) * drift_strength  # (W,)
    base = base + drift_direction[:, None, :] * week_factors[None, :, None]
    base = np.clip(base, 0.01, None)
    base = base / base.sum(axis=2, keepdims=True)

    # Sample all (bin, week, position) at once — (B, W, P, 10) intermediate
    combined   = base[:, :, None, :] * pos_biases[None, None, :, :]
    combined  /= combined.sum(axis=3, keepdims=True)
    cumulative = np.cumsum(combined, axis=3)
    r          = rng.random((NUM_BINS, NUM_WEEKS, NUM_POSITIONS, 1))
    items_array = (r < cumulative).argmax(axis=3).astype(np.int16)  # (B, W, P)

    return {
        'items':       items_array,
        'bin_ranks':   bin_ranks,
        'bin_segments': bin_segments,
        'bin_names':   np.array(BIN_NAMES),
        'dates':       dates,
        'item_codes':  list(ITEMS),
        # First 10 items get distinct colors; KLP/LMR/NQT are "other" (gray).
        'item_colors': list(COLORS[:N_MAX_USER_ITEMS]) + [OTHER_COLOR] * (len(ITEMS) - N_MAX_USER_ITEMS),
    }


@st.cache_data
def discover_items(file_bytes: bytes, filename: str):
    """Fast first pass: parse only enough to return item vocabulary and counts.
    Returns (items_by_freq, counts_dict) — items sorted descending by frequency."""
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), usecols=['item'])
    except Exception:
        return [], {}
    df['item'] = df['item'].astype(str)
    vc = df['item'].value_counts()
    return vc.index.tolist(), vc.to_dict()


@st.cache_data
def load_user_data(file_bytes: bytes, filename: str):
    """Parse an uploaded CSV. All items are kept; colour assignment happens in the UI."""
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        st.error(f"Could not parse CSV: {exc}")
        return None

    required = {'bin_id', 'date', 'position', 'item', 'bin_rank', 'segment'}
    missing  = required - set(df.columns)
    if missing:
        st.error(f"CSV is missing columns: {', '.join(sorted(missing))}")
        return None

    df['date']   = pd.to_datetime(df['date']).dt.date
    df['item']   = df['item'].astype(str)
    df['bin_id'] = df['bin_id'].astype(str)
    df['segment'] = df['segment'].astype(str)

    # Composite key: a bin_id that appears in multiple segments becomes distinct rows
    df['bin_key'] = df['bin_id'] + ' · ' + df['segment']

    # All items kept; order by descending frequency so callers can slice for colours
    user_items = df['item'].value_counts().index.tolist()

    def _majority_or_random(x):
        modes = x.mode()
        return modes.iloc[np.random.randint(len(modes))]

    # Collapse multiple measurements per (bin_key, date, position): majority item, random tiebreak
    key_cols = ['bin_key', 'date', 'position']
    dup_counts = df.groupby(key_cols).size()
    n_dup_keys = int((dup_counts > 1).sum())
    if n_dup_keys:
        df = (df.groupby(key_cols, as_index=False)
                .agg(item=('item', _majority_or_random),
                     bin_rank=('bin_rank', 'first'),
                     segment=('segment', 'first')))

    bin_keys  = sorted(df['bin_key'].unique())
    dates     = sorted(df['date'].unique())
    positions = sorted(df['position'].unique())

    n_bins  = len(bin_keys)
    n_weeks = len(dates)
    n_pos   = len(positions)

    item_to_idx  = {code: i for i, code in enumerate(user_items)}
    bin_idx_map  = {b: i for i, b in enumerate(bin_keys)}
    date_idx_map = {d: i for i, d in enumerate(dates)}
    pos_idx_map  = {p: i for i, p in enumerate(positions)}

    items_array = np.full((n_bins, n_weeks, n_pos), -1, dtype=np.int16)  # -1 = no data
    _bi = df['bin_key'].map(bin_idx_map).astype(np.intp).values
    _di = df['date'].map(date_idx_map).astype(np.intp).values
    _pi = df['position'].map(pos_idx_map).astype(np.intp).values
    _ii = df['item'].map(item_to_idx).fillna(-1).astype(np.int16).values
    items_array[_bi, _di, _pi] = _ii

    bin_meta = (
        df.drop_duplicates('bin_key')
        .set_index('bin_key')
        .loc[bin_keys, ['bin_rank', 'segment']]
    )

    return {
        'items':       items_array,
        'bin_ranks':   bin_meta['bin_rank'].to_numpy().astype(int),
        'bin_segments': bin_meta['segment'].to_numpy().astype(str),
        'bin_names':   np.array(bin_keys),
        'dates':       list(dates),
        'item_codes':  user_items,   # all items, frequency-ordered; no colours here
        'n_dup_keys':  n_dup_keys,
    }


# ============ HELPERS ============
def dim_color(hex_color, dim_amount=0.88, bg=BG):
    """Blend hex_color toward background by dim_amount (0=no dim, 1=full bg)."""
    fg     = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    bg_rgb = tuple(int(bg[i:i+2], 16) for i in (1, 3, 5))
    blended = tuple(int(fg[i] * (1 - dim_amount) + bg_rgb[i] * dim_amount) for i in range(3))
    return f'#{blended[0]:02x}{blended[1]:02x}{blended[2]:02x}'


def compute_majority(items_subset, n_items=10):
    """
    items_subset shape: (n_bins, n_weeks, n_positions)
    Returns (majority shape (n_bins, n_positions), share shape (n_bins, n_positions))
    Pure mode with most-recent tiebreak.
    """
    n_bins, n_weeks, n_positions = items_subset.shape
    if n_weeks == 1:
        result = items_subset[:, 0, :].astype(np.int16)
        return result, (result >= 0).astype(float)

    # Count per (bin, pos, item) without looping over n_items.
    # np.add.at visits each valid entry once — O(n_valid), not O(n_items × array_size).
    counts = np.zeros((n_bins, n_positions, n_items), dtype=np.int32)
    _b, _w, _p = np.where(items_subset >= 0)
    np.add.at(counts, (_b, _p, items_subset[_b, _w, _p].astype(np.intp)), 1)

    max_counts = counts.max(axis=2)
    majority   = counts.argmax(axis=2).astype(np.int16)
    no_data    = max_counts == 0
    majority[no_data] = -1

    most_recent  = items_subset[:, -1, :]
    b_idx, p_idx = np.meshgrid(np.arange(n_bins), np.arange(n_positions), indexing='ij')
    safe_recent  = np.clip(most_recent, 0, n_items - 1)
    most_recent_count = np.where(most_recent >= 0, counts[b_idx, p_idx, safe_recent], 0)
    majority = np.where(
        (most_recent >= 0) & (most_recent_count == max_counts),
        most_recent, majority,
    ).astype(np.int16)
    majority[no_data] = -1
    return majority, np.where(no_data, 0.0, max_counts / n_weeks)


def apply_url_params(dates, item_codes=None):
    """On a fresh session, seed widget keys from URL query params."""
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
        st.session_state['sort_radio'] = p['sort']  # guard below resets if invalid

    if 'cs' in p and 'cell_sz' not in st.session_state:
        try:
            st.session_state['cell_sz'] = max(6, min(28, int(p['cs'])))
        except ValueError:
            pass

    if 'af' in p and 'auto_fit_cb' not in st.session_state:
        st.session_state['auto_fit_cb'] = (p['af'] == '1')


def make_view_csv(bin_names, positions, items_grid, share_grid, ranks, segments,
                  item_codes=None, bin_term='bin'):
    if item_codes is None:
        item_codes = ITEMS
    n_bins, n_pos = items_grid.shape
    codes = np.array(item_codes)
    flat  = items_grid.ravel().astype(int)
    return pd.DataFrame({
        bin_term:         np.repeat(bin_names, n_pos),
        'rank':           np.repeat(ranks, n_pos).astype(int),
        'segment':        np.repeat(segments, n_pos),
        'position':       np.tile(positions.astype(int), n_bins),
        'item':           np.where(flat >= 0, codes[np.clip(flat, 0, len(codes) - 1)], ''),
        'majority_share': np.round(share_grid.ravel().astype(float), 4),
    }).to_csv(index=False).encode()


# ============ APP ============
# width='content'/'stretch' was introduced in Streamlit ~1.51; older versions
# require use_container_width. Detect once so both envs get correct behaviour.
_st_ver = tuple(int(x) for x in st.__version__.split('.')[:2])
_chart_own_width  = {'width': 'content'}  if _st_ver >= (1, 51) else {'use_container_width': False}
_chart_full_width = {'width': 'stretch'}  if _st_ver >= (1, 51) else {'use_container_width': True}
_btn_full_width   = {'width': 'stretch'}  if _st_ver >= (1, 51) else {'use_container_width': True}

st.set_page_config(
    page_title="Ranked Placement Atlas",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Read display customisation from session state before CSS is injected
_custom_bg         = st.session_state.get('bg_color', BG) or BG
_title_font_key    = st.session_state.get('title_font', 'Fraunces')
_title_font_css, _title_font_style = TITLE_FONTS.get(_title_font_key, TITLE_FONTS['Fraunces'])
_custom_title      = st.session_state.get('custom_title', '') or 'Ranked Placement Atlas'

# Custom styling
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
    /* all/none/sort-guide: link-style utility — no border, plain text */
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
    /* tighten gap + column padding for nested column rows (filter headers) */
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
    /* ── Filter-row divider spacing ── */
    /* Streamlit 1.57: st.divider() renders as hr inside stMarkdownContainer.
       Default: hr has margin 32px top/bottom; parent has margin-bottom -16px.
       Override hr margins on main content only (not sidebar/dialogs). */
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] hr {{
        margin-top:    2px !important;
        margin-bottom: 2px !important;
    }}
    [data-testid="stMain"] [data-testid="stMarkdownContainer"]:has(hr) {{
        margin-bottom: 0 !important;
    }}
    /* Tighten the gap between all top-level blocks in the main content stack */
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
        help="Required columns: bin_id, date, position, item, bin_rank, segment.",
        label_visibility="collapsed",
    )
    colored_items = None   # set below when file is uploaded
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
                format_func=lambda x: f"{x}  ({item_counts[x]:,})",
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
            n_b, n_w, n_p = data['items'].shape
            n_it = len(data['item_codes'])
            st.success(
                f"{uploaded.name}\n\n"
                f"{n_b} bins · {n_p} positions · {n_w} snapshots · {n_it} items"
            )
            _n_dup = data.get('n_dup_keys', 0)
            if _n_dup:
                st.warning(
                    f"{_n_dup:,} (bin, date, position) combinations had multiple rows "
                    f"— kept the most frequent item for each."
                )
    else:
        st.caption("Using synthetic demo data. Upload a CSV to use your own data.")
        st.caption(
            "Required columns: `bin_id`, `date`, `position`, `item`, `bin_rank`, `segment`  \n"
            f"Up to {N_MAX_USER_ITEMS} items get distinct colours; extras shown in gray.  \n"
            "`bin_id` is used as the display name."
        )

    st.divider()
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:11px;'
        f'letter-spacing:0.15em;text-transform:uppercase;color:{INK};'
        f'margin-bottom:8px;">Labels</div>',
        unsafe_allow_html=True,
    )
    bin_term    = st.text_input("Bins are called",    "bin",    key="bin_term").strip()    or "bin"
    item_term   = st.text_input("Items are called",   "item",   key="item_term").strip()   or "item"
    segment_term = st.text_input(f"{bin_term.capitalize()} grouping attribute", "segment", key="segment_term").strip() or "segment"

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
    # Placeholder filled later (after _new_params is built) so the URL
    # always reflects the CURRENT run's filter state, not the previous run's.
    try:
        _share_host  = st.context.headers.get("host", "localhost:8502")
        _share_proto = "https" if not _share_host.startswith("localhost") else "http"
    except Exception:
        _share_host, _share_proto = "localhost:8502", "http"
    _share_placeholder = st.empty()
    st.caption("Link encodes your current filters, sort mode, and date range.")

# ── Extract item vocabulary and threading variables ───────────────────────────
item_codes = data.get('item_codes', list(ITEMS))
n_items    = len(item_codes)

# Build item_colors: colored_items get distinct COLORS, everything else gets OTHER_COLOR.
if colored_items is not None:
    _colored_set = set(colored_items)
    _color_idx   = {item: i for i, item in enumerate(colored_items)}
    item_colors  = [
        COLORS[_color_idx[item]] if item in _colored_set else OTHER_COLOR
        for item in item_codes
    ]
    # pill_items: only the distinctly-coloured items shown in the pills
    pill_items = list(colored_items)
else:
    item_colors = data.get('item_colors', list(COLORS[:n_items]))
    # Extend to n_items if the stored colors list is shorter than the vocabulary.
    if len(item_colors) < n_items:
        item_colors = item_colors + [OTHER_COLOR] * (n_items - len(item_colors))
    # Items assigned OTHER_COLOR are "other" — exclude from interactive pills
    # so that n_gray > 0 and the JS-injected "other (N)" pill appears.
    pill_items = [c for c, col in zip(item_codes, item_colors) if col != OTHER_COLOR]
    if not pill_items:
        pill_items = list(item_codes)

# Reset items_pills when the pill options change (new file or new colour selection)
_item_sig = ','.join(pill_items)
if st.session_state.get('_item_sig') != _item_sig:
    st.session_state.pop('items_pills', None)
    st.session_state['_item_sig'] = _item_sig

available_segments = sorted(np.unique(data['bin_segments']).tolist())

# Reset segments_pills when the segment vocabulary changes (e.g., new CSV uploaded)
_segment_sig = ','.join(available_segments)
if st.session_state.get('_segment_sig') != _segment_sig:
    st.session_state.pop('segments_pills', None)
    st.session_state['_segment_sig'] = _segment_sig

# Per-item pill colors via JS — only pill_items get distinct colours.
# Discriminator: skip button groups that contain a segment label.
_pill_colors  = [item_colors[item_codes.index(it)] for it in pill_items]
_n_gray       = n_items - len(pill_items)
# "other (N)" is added as a real st.pills entry below; include its gray colour so
# the JS colorizer styles it correctly.  JS injection (NGRAY) is disabled.
_other_pill   = f'other ({_n_gray})' if _n_gray > 0 else None
_colors_json  = json.dumps(_pill_colors + ([OTHER_COLOR] if _other_pill else []))
_segments_json = json.dumps(available_segments)
st.html(
    f"""<script>
(function(){{
  var C={_colors_json}, BG="{_custom_bg}", N=C.length, R={_segments_json}, NGRAY=0, GRAY="{OTHER_COLOR}";
  function go(){{
    try{{
      var gs=document.querySelectorAll('[data-baseweb="button-group"]');
      for(var i=0;i<gs.length;i++){{
        var bs=gs[i].querySelectorAll('button');
        // Exclude any previously-injected gray pills from the count
        var realBs=Array.from(bs).filter(function(b){{return !b.getAttribute('data-gray-pill');}});
        if(realBs.length!==N) continue;
        // Skip the segments group (contains known segment labels)
        var texts=realBs.map(function(b){{return b.textContent.trim();}});
        if(R.some(function(r){{return texts.indexOf(r)>=0;}})) continue;
        // Color the real pills
        for(var j=0;j<realBs.length;j++){{
          var b=realBs[j];
          var sel=b.getAttribute('aria-pressed')==='true'||b.getAttribute('aria-checked')==='true';
          b.style.setProperty('color', sel ? BG : C[j], 'important');
          b.style.setProperty('border-color', C[j], 'important');
          b.style.setProperty('background-color', sel ? C[j] : '', 'important');
        }}
        // Inject "other (N)" pill as a SIBLING after the button-group, not inside it.
        // Appending inside the React-managed button-group gets reconciled away immediately.
        // A sibling element survives React's reconciliation of its children.
        if(NGRAY>0){{
          var sib=gs[i].nextElementSibling;
          if(!sib || sib.getAttribute('data-gray-pill')!=='1'){{
            // Remove any stale gray pill elsewhere under the same parent.
            var stale=gs[i].parentNode.querySelector('[data-gray-pill]');
            if(stale) stale.parentNode.removeChild(stale);
            var rs=window.getComputedStyle(realBs[0]);
            gs[i].insertAdjacentHTML('afterend',
              '<button data-gray-pill="1" style="'
              +'height:'+rs.height+';padding:'+rs.paddingTop+' '+rs.paddingRight+' '+rs.paddingBottom+' '+rs.paddingLeft+';'
              +'font-family:'+rs.fontFamily+';font-size:'+rs.fontSize+';'
              +'border-radius:'+rs.borderRadius+';'
              +'background-color:'+GRAY+';color:#fff;border:1px solid '+GRAY+';'
              +'opacity:0.7;cursor:default;pointer-events:none;margin-left:4px;vertical-align:middle;white-space:nowrap;'
              +'">'+'other ('+NGRAY+')'+'</button>'
            );
          }}
        }}
      }}
    }}catch(e){{}}
  }}
  go();
  try{{
    new MutationObserver(go).observe(
      document.body,
      {{subtree:true,childList:true,attributes:true,attributeFilter:['aria-pressed','aria-checked']}}
    );
  }}catch(e){{ setInterval(go,200); }}
}})();
</script>"""
)

# Apply URL query params to session state (only on fresh sessions)
apply_url_params(data['dates'], item_codes)

n_pos_total  = data['items'].shape[2]
min_rank_val = int(data['bin_ranks'].min())
max_rank_val = int(data['bin_ranks'].max())

# ── User guide dialog ─────────────────────────────────────────────────────────
@st.dialog("User Guide", width="large")
def _show_user_guide():
    st.markdown(f"""
### What you're looking at

A heatmap where each **row** is a {bin_term}, each **column** is a ranked position,
and each **cell** is coloured by the {item_term} occupying that slot.
When you select a date range spanning multiple snapshots, each cell shows the
**most frequent {item_term}** across those dates (ties broken by the most recent snapshot).
Hover any cell for exact values.

Below the heatmap is a stacked bar showing, for each position, what share of
**visible {bin_term}s have that {item_term} as their majority winner** — a count of
wins, not a share-weighted average. A {bin_term} where an {item_term} won with 35%
and one where it won with 95% each contribute equally (one count each).

---

### Filters — Row 1

**{segment_term.capitalize()}s** · Toggleable pills. Selecting a subset hides {bin_term}s whose {segment_term} is not selected.
Use **all** / **none** to select or clear in one click.

**{item_term.capitalize()}s** · Toggleable pills. Selection controls *highlighting*, not filtering —
all {item_term}s remain visible, but unselected ones are dimmed.
Use **all** / **none** to reset or clear.

---

### Ranges — Row 2

**Date** · Dual-handle slider over all snapshots in the data.
Narrowing to a single date shows that exact snapshot; a wider range
triggers the majority metric described above.

**{bin_term.capitalize()} rank range** · Filter rows by global rank.
Drag either handle to zoom in on top- or bottom-ranked {bin_term}s.

**Position range** · Hide columns outside the chosen window.
Sort keys are always computed over the full position set, so row
order stays stable as you narrow the window.

---

### Sort — Row 3

| Mode | What it does |
|---|---|
| **Index** | Original data order — unsorted baseline |
| **Similarity** | Groups {bin_term}s that share the same {item_term}s at positions 1–4 (default) |
| **{bin_term.capitalize()} Rank** | Ascending by global rank — top-ranked {bin_term}s at top |
| **Top-rank** | Groups {bin_term}s that share the same {item_term} at position 1; ties resolved by position 2, then 3, and so on |
| **Selected Share** | Ranks {bin_term}s by how many visible positions are held by selected {item_term}s; ties broken by which {bin_term} has them at earlier positions (available when 1 – N−1 items are highlighted) |

[Visual guide to all sort modes →]({SORT_GUIDE_URL})

---

### Cell size & Export

**Cell size** slider and **Auto-fit** checkbox sit just below the View Summary block,
next to the download buttons. Auto-fit expands cells to use available space;
the slider sets a minimum (and the exact size when Auto-fit is off).

**Download CSV** exports the current view (visible {bin_term}s, positions, and date range).
**Export heatmap** serialises the interactive Plotly chart to a self-contained HTML file you can share or archive.

---

### Drill-down

Click any cell in the heatmap to select that {bin_term}.
A time-series heatmap for that {bin_term} appears below the main chart,
showing how its position-by-position {item_term} mix evolved over all dates.

---

### Display customisation (sidebar)

Open the **sidebar** (arrow at top-left) to find the **Display** section:

- **Title** — change the heading text to match your domain or presentation.
- **Title font** — choose from four options: *Fraunces* (italic serif, default), *Playfair Display* (classic serif), *DM Serif Display* (modern serif), or *IBM Plex Sans* (clean sans-serif).
- **Background color** — pick any colour; the page background, sidebar, chart area, and empty-cell colour all update together.

---

### Uploading your own data

Open the **sidebar** (arrow at top-left) and upload a CSV with these columns:

| Column | Notes |
|---|---|
| `bin_id` | Display name for the {bin_term} |
| `date` | Any format `pd.to_datetime` accepts |
| `position` | Integer rank within the {bin_term} (1-based or 0-based) |
| `item` | Any string label |
| `bin_rank` | Global rank of the {bin_term} |
| `segment` | Grouping / filter attribute — rename via **Labels → {bin_term.capitalize()} grouping attribute** |

If the same `bin_id` appears with multiple `segment` values, each `(bin_id, segment)`
pair becomes a distinct row labelled `bin_id · segment`. Multiple rows for the same (bin, date, position)
are resolved by majority vote (random tiebreak), producing **one item per date slot**.
When you then select a date range, each date casts one vote — raw row counts do not carry forward.

Up to **11** {item_term}s can receive distinct colours; the rest render in gray
but still show their real label on hover and in cell text.
""")

# ── Sort guide dialog ─────────────────────────────────────────────────────────
@st.dialog("Sort modes — visual guide", width="large")
def _show_sort_guide():
    try:
        html = Path(__file__).with_name("sort_modes_explainer.html").read_text(encoding="utf-8")
        style_m = re.search(r'<style>(.*?)</style>', html, re.DOTALL)
        body_m  = re.search(r'<body>(.*?)</body>',  html, re.DOTALL)
        if style_m and body_m:
            st.html(f"<style>{style_m.group(1)}</style>{body_m.group(1)}")
        else:
            st.html(html)
    except FileNotFoundError:
        st.markdown(f"[Open visual guide in browser →]({SORT_GUIDE_URL})")

# ── Title ─────────────────────────────────────────────────────────────────────
_title_col, _help_col = st.columns([9, 1])
with _title_col:
    st.markdown(f"""
<div class="title-block">
    <div class="title" style="font-family:{_title_font_css};font-style:{_title_font_style};">{_custom_title}</div>
    <div class="subtitle">{len(data['bin_names'])} {bin_term}s &times; {n_pos_total} positions &times; {n_items} {item_term}s &times; {len(data['dates'])} snapshots. When multiple dates are selected, each cell shows the most frequent {item_term} at that position across the selected dates — ties broken by the most recent snapshot.</div>
</div>
""", unsafe_allow_html=True)
with _help_col:
    st.write("")
    st.write("")
    if st.button("User guide", key="help_btn"):
        _show_user_guide()

# ── Row 1: Filters — Regions + Items ──────────────────────────────────────────

col_segments, col_items = st.columns(2)

with col_segments:
    if 'segments_pills' in st.session_state:
        stored = st.session_state['segments_pills']
        valid = [r for r in stored if r in available_segments]
        if stored and not valid:
            del st.session_state['segments_pills']  # dataset changed, reset
        else:
            st.session_state['segments_pills'] = valid  # preserve empty (None clicked)

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
    with _sall:
        st.button("all",  key="btn_reg_all",  on_click=_reg_all)
    with _snone:
        st.button("none", key="btn_reg_none", on_click=_reg_none)
    selected_segments = st.pills(
        f"{segment_term.capitalize()}s",
        available_segments,
        selection_mode="multi",
        default=available_segments,
        key="segments_pills",
        label_visibility="collapsed",
    )

with col_items:
    if 'items_pills' not in st.session_state:
        st.session_state['items_pills'] = list(pill_items)

    _all_items = list(pill_items)
    def _items_all():  st.session_state['items_pills'] = _all_items
    def _items_none(): st.session_state['items_pills'] = []
    # on_change fires at the start of the next run (before the widget is
    # instantiated), so session state is still writable at that point.
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
    with _iall:
        st.button("all",  key="btn_all",   on_click=_items_all)
    with _inone:
        st.button("none", key="btn_clear", on_click=_items_none)
    # Append "other (N)" as a real (but non-actionable) pill so React manages it.
    _pill_display = pill_items + ([_other_pill] if _other_pill else [])
    selected_items = st.pills(
        f"{item_term.capitalize()}s",
        _pill_display,
        selection_mode="multi",
        key="items_pills",
        on_change=_strip_other,
        label_visibility="collapsed",
    )
    # Also filter from the value returned this run (covers the click cycle).
    if _other_pill:
        selected_items = [s for s in (selected_items or []) if s != _other_pill]

# ── Row 2: Ranges — Date / Bin Rank / Position ────────────────────────────────
st.divider()

col_date, col_rank, col_pos = st.columns(3)

with col_date:
    date_range = st.select_slider(
        "Date",
        options=data['dates'],
        value=(data['dates'][-13], data['dates'][-1]),
        format_func=lambda d: d.strftime("%b %d, '%y"),
        key="wk_slider",
    )

with col_rank:
    _rk = st.session_state.get('rank_slider')
    if _rk is not None and (_rk[0] < min_rank_val or _rk[1] > max_rank_val or _rk[0] > _rk[1]):
        del st.session_state['rank_slider']
    rank_range = st.slider(
        f"{bin_term.capitalize()} rank range", min_rank_val, max_rank_val, (min_rank_val, max_rank_val),
        key="rank_slider",
    )

with col_pos:
    pos_range = st.slider(
        "Position range", 1, n_pos_total, (1, n_pos_total),
        key="pos_slider",
    )

st.divider()

# ── Row 3: Sort (full width) ───────────────────────────────────────────────────
n_sel        = len(selected_items) if selected_items else 0
n_pill_items = len(pill_items)

sort_options = ["Index", "Similarity", f"{bin_term.capitalize()} Rank", "Top-rank"]
if 0 < n_sel < n_pill_items:
    sort_options.append("Selected Share")

if st.session_state.get('sort_radio') not in sort_options:
    st.session_state['sort_radio'] = 'Similarity'

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
    index=sort_options.index(st.session_state.get('sort_radio', 'Similarity')),
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
segments_active = selected_segments if selected_segments else available_segments

in_segment = np.isin(data['bin_segments'], segments_active)
in_rank   = (data['bin_ranks'] >= rank_range[0]) & (data['bin_ranks'] <= rank_range[1])
visible_mask        = in_segment & in_rank
visible_bin_indices = np.where(visible_mask)[0]

date_start_idx = data['dates'].index(date_range[0])
date_end_idx   = data['dates'].index(date_range[1])
date_indices   = list(range(date_start_idx, date_end_idx + 1))

pos_indices = list(range(pos_range[0] - 1, pos_range[1]))

if len(visible_bin_indices) == 0 or len(date_indices) == 0 or len(pos_indices) == 0:
    st.warning("No data in current filter range. Widen your selectors.")
    st.stop()

# ── Majority computation ───────────────────────────────────────────────────────
items_sub = data['items'][visible_bin_indices, date_start_idx:date_end_idx + 1, :]
majority, share = compute_majority(items_sub, n_items)   # (n_vis, n_pos_total)

majority_f = majority[:, pos_indices]
share_f    = share[:, pos_indices]

# ── Sort ──────────────────────────────────────────────────────────────────────
n_vis = len(visible_bin_indices)
majority_sort = majority

if sort_mode == "Index":
    order = np.arange(n_vis)
elif sort_mode == f"{bin_term.capitalize()} Rank":
    order = np.argsort(data['bin_ranks'][visible_bin_indices], kind='stable')
elif sort_mode == "Similarity":
    top4  = majority_sort[:, :4].astype(np.int32)
    order = np.lexsort(top4.T[::-1])   # primary key = col 0, secondary = col 1, …
elif sort_mode == "Top-rank":
    order = np.lexsort(majority_sort.astype(np.int32).T[::-1])
elif sort_mode == "Selected Share":
    sel_idx_set  = [item_codes.index(i) for i in selected_items]
    sel_mask     = np.isin(majority_f, sel_idx_set)
    share_count  = sel_mask.sum(axis=1)
    pos_sum      = (sel_mask * (np.array(pos_indices) + 1)).sum(axis=1)   # lower = more prominent
    order        = np.lexsort([pos_sum, -share_count])        # primary: share; secondary: position
else:
    order = np.arange(n_vis)

ordered_bin_indices = visible_bin_indices[order]
majority_disp       = majority_f[order]
share_disp          = share_f[order]

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

# Colorscale spans [-1, n_items]: slot 0 = no-data (background), slots 1..n_items = items.
_n = n_items + 1
colorscale = [[0.0, _custom_bg], [1 / _n, _custom_bg]]
for i in range(n_items):
    colorscale.append([(i + 1) / _n, _eff_colors[i]])
    colorscale.append([(i + 2) / _n, _eff_colors[i]])

# ── Sizing (widgets rendered below view summary; read from session state here) ─
cell_size = st.session_state.get('cell_sz', 12)
auto_fit  = st.session_state.get('auto_fit_cb', True)

# ── Sizing ────────────────────────────────────────────────────────────────────
container_w = 900
container_h = 720

if auto_fit:
    cell_w = max(cell_size, container_w / max(n_show_pos, 1))
    cell_h = max(cell_size, container_h / max(n_show_bins, 1))
else:
    cell_w = cell_size
    cell_h = cell_size

cell_w = min(cell_w, 40)
cell_h = min(cell_h, 30)

heatmap_width  = int(n_show_pos * cell_w)
heatmap_height = int(n_show_bins * cell_h)
total_width    = heatmap_width + 170
total_height   = int(heatmap_height / 0.83) + 60

# ── Display arrays ────────────────────────────────────────────────────────────
positions_disp = np.array(pos_indices) + 1
ranks_disp     = data['bin_ranks'][ordered_bin_indices]
segments_disp   = data['bin_segments'][ordered_bin_indices]
bin_names_disp = data['bin_names'][ordered_bin_indices]

# bin_name → %{y}, position → %{x} in hovertemplate; no need to repeat per cell.
# 3-column customdata: rank (per-row), segment (per-row), share (per-cell).
customdata = np.empty((n_show_bins, n_show_pos, 3), dtype=object)
customdata[:, :, 0] = ranks_disp[:, None].astype(int)
customdata[:, :, 1] = segments_disp[:, None]
customdata[:, :, 2] = share_disp.astype(float)

_codes    = np.array(item_codes)
text_grid = np.where(
    majority_disp >= 0,
    _codes[np.clip(majority_disp.astype(int), 0, n_items - 1)],
    '—',
)
z         = majority_disp.astype(float)

# ── View summary ──────────────────────────────────────────────────────────────
date_count = len(date_indices)
multi_date = date_count > 1

if multi_date:
    d0 = date_range[0].strftime("%b %d, %Y")
    d1 = date_range[1].strftime("%b %d, %Y")
    mode_sentence = (
        f"Each cell shows the <b>most frequent {item_term}</b> across the {date_count} selected snapshots "
        f"({d0} → {d1}), with ties broken by the most recent snapshot. "
        f"Hover to see the majority share — how often that {item_term} actually held the position."
    )
    _date_label = f"{date_count} snapshots"
else:
    d0 = date_range[0].strftime("%b %d, %Y")
    mode_sentence = f"Showing a single snapshot ({d0}): each cell is the {item_term} at that position."
    _date_label = "1 snapshot"

summary_html = f"""
<div style="background:rgba(0,0,0,0.04);border-radius:4px;
            padding:12px 16px;margin-bottom:8px;max-width:900px;">
    <p style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;
              font-weight:600;color:{INK};margin:0 0 6px 0;">Showing</p>
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

# ── Legend (below summary, directly above heatmap) ────────────────────────────
_item_code_idx = {code: i for i, code in enumerate(item_codes)}
legend_parts = []
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
st.markdown(
    '<div style="display:flex;flex-wrap:wrap;gap:2px;margin-top:10px;margin-bottom:4px;">'
    + ''.join(legend_parts) + '</div>',
    unsafe_allow_html=True,
)

csv_bytes = make_view_csv(
    bin_names_disp, positions_disp, majority_disp, share_disp,
    ranks_disp, segments_disp, item_codes=item_codes, bin_term=bin_term,
)

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
        zmax=n_items,
        showscale=False,
        customdata=customdata,
        text=text_grid,
        hovertemplate=(
            "<b>%{y}</b>  ·  Rank %{customdata[0]}  ·  %{customdata[1]}<br>"
            f"Position %{{x}}  ·  {item_term.capitalize()} <b>%{{text}}</b><br>"
            "Majority share: %{customdata[2]:.0%}"
            "<extra></extra>"
        ),
        xgap=0.5,
        ygap=0.5,
    ),
    row=1, col=1,
)

# Split items into distinctly-coloured vs gray "other" to cap trace count at ~12,
# regardless of how many unique items are in the uploaded vocabulary.
_colored_idx = [i for i in range(n_items) if item_colors[i] != OTHER_COLOR]
_other_idx   = [i for i in range(n_items) if item_colors[i] == OTHER_COLOR]

_n_bins_safe = max(n_show_bins, 1)
# Loop bounded by number of distinct colors (≤ len(COLORS)), not vocabulary size.
for local_i, item_idx in enumerate(_colored_idx):
    fig.add_trace(
        go.Bar(
            x=positions_disp,
            y=(majority_disp == item_idx).sum(axis=0) / _n_bins_safe,
            orientation='v',
            marker=dict(color=_eff_colors[item_idx], line=dict(width=0)),
            name=item_codes[item_idx],
            showlegend=False,
            hovertemplate=f"<b>{item_codes[item_idx]}</b> is majority winner in %{{y:.0%}} of {bin_term}s · pos %{{x}}<extra></extra>",
        ),
        row=2, col=1,
    )

if _other_idx:
    # Aggregate all "other" items into one gray bar trace.
    _other_count = np.isin(majority_disp, _other_idx).sum(axis=0) / _n_bins_safe
    fig.add_trace(
        go.Bar(
            x=positions_disp,
            y=_other_count,
            orientation='v',
            marker=dict(color=OTHER_COLOR, line=dict(width=0)),
            name='Other',
            showlegend=False,
            hovertemplate=f"<b>Other</b> is majority winner in %{{y:.0%}} of {bin_term}s · pos %{{x}}<extra></extra>",
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
        bgcolor=INK,
        bordercolor=INK,
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
    title=dict(text=f'% of {bin_term}s · majority winner', font=dict(size=9, family='IBM Plex Mono', color=MUTED)),
    row=2, col=1,
)

# ── Render chart ──────────────────────────────────────────────────────────────
chart_event = st.plotly_chart(
    fig,
    **_chart_own_width,
    on_select="rerun",
    key="main_chart",
)

# ── HTML export — cached in session state; only reserialised when view changes ──
_html_view_key = (
    tuple(ordered_bin_indices.tolist()),
    date_start_idx, date_end_idx,
    tuple(pos_indices),
    sort_mode,
    tuple(sorted(sel_idx_set)),
    _custom_bg,
)
# Lazy HTML export: invalidate when the view changes but don't re-serialise until
# the user explicitly requests it.  fig.to_html() can be expensive for large figures.
if st.session_state.get('_html_view_key') != _html_view_key:
    st.session_state['_html_view_key'] = _html_view_key
    st.session_state.pop('_html_bytes', None)   # clear stale bytes

# ── Chart footer: cell size + auto-fit + downloads (all output controls) ─────
_sz_col, _af_col, _gap_col, _dl_csv_col, _dl_html_col = st.columns([3, 1, 1, 1, 1])
with _sz_col:
    st.slider("Cell size (px)", 6, 28, cell_size, key="cell_sz")
with _af_col:
    st.write("")
    st.checkbox("Auto-fit", value=auto_fit, key="auto_fit_cb")
with _dl_csv_col:
    st.download_button(
        "Download CSV",
        csv_bytes,
        file_name="atlas_view.csv",
        mime="text/csv",
        **_btn_full_width,
    )
with _dl_html_col:
    if '_html_bytes' in st.session_state:
        st.download_button(
            "Download heatmap",
            st.session_state['_html_bytes'],
            file_name="atlas_view.html",
            mime="text/html",
            **_btn_full_width,
        )
    elif st.button("Export heatmap", **_btn_full_width):
        st.session_state['_html_bytes'] = fig.to_html(
            include_plotlyjs='cdn', config={'displayModeBar': True}
        ).encode()
        st.rerun()

# ── Drill-down ────────────────────────────────────────────────────────────────
# Detect clicked bin from chart event (heatmap trace is curve 0)
clicked_bin_name = None
if chart_event and chart_event.selection and chart_event.selection.points:
    for pt in chart_event.selection.points:
        if pt.get('curve_number', -1) == 0 and pt.get('y'):
            clicked_bin_name = pt['y']
            break

drill_col, _ = st.columns([2, 3])
with drill_col:
    _no_sel = f"— select a {bin_term} —"
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
        bidx        = bin_matches[0]
        bin_rank_v  = data['bin_ranks'][bidx]
        bin_segment  = data['bin_segments'][bidx]
        bin_weekly  = data['items'][bidx]               # (n_weeks, n_pos_total)
        drill_items = bin_weekly[date_indices][:, pos_indices]  # (n_dates, n_pos)
        drill_dates = data['dates'][date_start_idx:date_end_idx + 1]

        _segment_suffix = f" · {bin_segment}" if ' · ' not in drill_bin else ""
        with st.expander(
            f"Time series · {drill_bin} · Rank {bin_rank_v}{_segment_suffix}",
            expanded=True,
        ):
            mini_z    = drill_items.T.astype(float)           # (n_pos, n_dates)
            _di       = drill_items.T.astype(int)
            mini_text = np.where(_di >= 0, _codes[np.clip(_di, 0, n_items - 1)], '—')
            x_labels  = [d.strftime("%b %d") for d in drill_dates]
            y_labels  = list(positions_disp)

            mini_fig = go.Figure(go.Heatmap(
                z=mini_z,
                x=x_labels,
                y=y_labels,
                colorscale=colorscale,
                zmin=-1, zmax=n_items,
                showscale=False,
                text=mini_text,
                hovertemplate=(
                    "Date: <b>%{x}</b>  ·  Position: <b>%{y}</b><br>"
                    f"{item_term.capitalize()}: <b>%{{text}}</b><extra></extra>"
                ),
                xgap=0.5, ygap=0.5,
            ))

            mini_cell_w = max(10, min(30, 700 // max(len(date_indices), 1)))
            mini_cell_h = max(6,  min(20, 400 // max(n_show_pos, 1)))
            mini_w = int(len(date_indices) * mini_cell_w) + 80
            mini_h = int(n_show_pos * mini_cell_h) + 60

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
            _n_di, _n_dp = drill_items.shape
            _d_flat = drill_items.ravel().astype(int)
            _codes  = np.array(item_codes)
            drill_csv = pd.DataFrame({
                bin_term:   drill_bin,
                'rank':     int(bin_rank_v),
                'segment':  bin_segment,
                'date':     np.repeat([d.isoformat() for d in drill_dates], _n_dp),
                'position': np.tile(positions_disp.astype(int), _n_di),
                'item':     np.where(_d_flat >= 0, _codes[np.clip(_d_flat, 0, n_items - 1)], ''),
            }).to_csv(index=False).encode()
            st.download_button(
                f"Download {drill_bin} time series CSV",
                drill_csv,
                file_name=f"{drill_bin.lower()}_timeseries.csv",
                mime="text/csv",
            )

# ── Update URL query params (reflects current state for sharing) ───────────────
# Only write when something actually changed — unconditional writes trigger a
# rerun loop in some Streamlit versions, causing infinite local loading.
_new_params = {
    'segments': ','.join(sorted(set(segments_active))),
    'items':   ','.join(selected_items) if selected_items else '',
    'ds':      date_range[0].isoformat(),
    'de':      date_range[1].isoformat(),
    'rk0':     str(rank_range[0]),
    'rk1':     str(rank_range[1]),
    'ps0':     str(pos_range[0]),
    'ps1':     str(pos_range[1]),
    'sort':    sort_mode,
    'cs':      str(cell_size),
    'af':      '1' if auto_fit else '0',
}
if dict(st.query_params) != _new_params:
    st.query_params.update(_new_params)

# ── Fill share-button placeholder with current-run URL ────────────────────────
# Done here (after _new_params) so the URL always reflects the current state.
_qs_share  = urllib.parse.urlencode(_new_params, doseq=True)
_share_url = f"{_share_proto}://{_share_host}/" + (f"?{_qs_share}" if _qs_share else "")
_url_js    = json.dumps(_share_url)
with _share_placeholder.container():
 st.iframe(srcdoc=f"""
<script>
function copyLink(){{
  var url={_url_js};
  var btn=document.getElementById('share-btn');
  function ok(){{
    btn.textContent='✓ Copied';
    btn.style.color='#22c55e';
    btn.style.borderColor='#22c55e';
    setTimeout(function(){{
      btn.textContent='Copy link to this view';
      btn.style.color='';
      btn.style.borderColor='';
    }},2000);
  }}
  function execCopy(){{
    var ta=document.createElement('textarea');
    ta.value=url;ta.style.cssText='position:fixed;opacity:0;';
    document.body.appendChild(ta);ta.focus();ta.select();
    var ok2=false;
    try{{ok2=document.execCommand('copy');}}catch(e){{}}
    document.body.removeChild(ta);
    if(ok2){{ok();}}else{{prompt('Copy URL:',url);}}
  }}
  if(navigator.clipboard&&navigator.clipboard.writeText){{
    navigator.clipboard.writeText(url).then(ok,execCopy);
  }}else{{
    execCopy();
  }}
}}
</script>
<button id="share-btn" onclick="copyLink()"
  style="font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:0.08em;
         text-transform:uppercase;border:1px solid #888;background:transparent;
         padding:6px 14px;cursor:pointer;color:#1A1A1A;width:100%;
         transition:color .15s,border-color .15s;">
  Copy link to this view
</button>""", height=40)

