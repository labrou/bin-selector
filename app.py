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
        - region:   one of NA, SA, EU, AS, CN, AU (constant per bin)
    Drift model:
        - Slow archetype drift across weeks
        - 20% of bins undergo a one-time regime change between weeks 15-37
"""

import io
import json
import time
import streamlit as st
import streamlit.components.v1 as components
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

ITEMS = ['APX', 'BRT', 'CFD', 'DLT', 'ETR', 'FRM', 'GVS', 'HXC', 'INV', 'JTL']
COLORS = ['#B91C1C', '#1E3A8A', '#15803D', '#CA8A04', '#6D28D9',
          '#DB2777', '#0E7490', '#525252', '#92400E', '#4D7C0F',
          '#C2410C', '#0369A1']  # two extra slots for uploaded data
REGIONS = ['NA', 'SA', 'EU', 'AS', 'CN', 'AU']
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
PANEL_BG = '#FBFAF5'
INK      = '#1A1A1A'
MUTED    = '#6B6B6B'

def sort_descriptions(bt, it):
    """Return sort-mode description strings using the active bin/item terminology."""
    return {
        "Index":          f"Alphabetical order of {bt} ID — no analytical grouping; stable baseline.",
        "Similarity":     f"{bt.capitalize()}s sharing the same {it}s at positions 1–4 cluster together, surfacing archetypes as broad horizontal color bands. Default.",
        f"{bt.capitalize()} Rank": f"Top = highest-ranked {bt}s (rank 1). Use this when the question is about rank: do top {bt}s share a distinct profile?",
        "Top-rank":       f"Full lexicographic sort over all positions. The leftmost column is perfectly grouped; later columns fragment.",
        "Selected Share": f"Top = {bt}s whose top-10 positions are most saturated by the selected {it}s. Only available when 1–9 {it}s are highlighted.",
    }

# ============ DATA GENERATION ============
@st.cache_data
def generate_data():
    rng = np.random.default_rng(42)

    archetypes = np.array([
        [0.32, 0.18, 0.14, 0.10, 0.08, 0.06, 0.04, 0.03, 0.03, 0.02],
        [0.04, 0.28, 0.22, 0.14, 0.10, 0.08, 0.06, 0.04, 0.02, 0.02],
        [0.08, 0.08, 0.08, 0.20, 0.18, 0.14, 0.10, 0.06, 0.04, 0.04],
        [0.04, 0.04, 0.06, 0.06, 0.08, 0.12, 0.16, 0.18, 0.14, 0.12],
        [0.14, 0.06, 0.16, 0.06, 0.14, 0.08, 0.14, 0.06, 0.10, 0.06],
    ])

    def positional_bias(pos):
        if pos < 5:
            return np.array([3.0, 2.4, 2.0, 1.2, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8])
        if pos < 15:
            return np.array([1.2, 1.2, 1.2, 2.0, 2.0, 1.8, 1.0, 1.0, 0.8, 0.8])
        return np.ones(10)

    pos_biases = np.array([positional_bias(p) for p in range(NUM_POSITIONS)])  # (50, 10)

    bin_archetypes  = rng.integers(0, 5, NUM_BINS)
    bin_regions     = np.array(rng.choice(REGIONS, NUM_BINS))
    archetype_rank_center = np.array([15, 35, 50, 70, 50])
    rank_noise      = (rng.random(NUM_BINS) + rng.random(NUM_BINS) + rng.random(NUM_BINS) - 1.5) * 18
    _raw_scores     = archetype_rank_center[bin_archetypes] + rank_noise
    bin_ranks       = (np.argsort(np.argsort(_raw_scores)) + 1).astype(int)  # dense 1..NUM_BINS

    has_regime      = rng.random(NUM_BINS) < 0.20
    regime_weeks    = rng.integers(15, 38, NUM_BINS)
    regime_new_arch = rng.integers(0, 5, NUM_BINS)
    drift_direction = rng.normal(0, 1, (NUM_BINS, 10))
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
        'bin_regions': bin_regions,
        'bin_names':   np.array(BIN_NAMES),
        'dates':       dates,
        'item_codes':  list(ITEMS),
        'item_colors': list(COLORS),
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

    required = {'bin_id', 'date', 'position', 'item', 'bin_rank', 'region'}
    missing  = required - set(df.columns)
    if missing:
        st.error(f"CSV is missing columns: {', '.join(sorted(missing))}")
        return None

    df['date']   = pd.to_datetime(df['date']).dt.date
    df['item']   = df['item'].astype(str)
    df['bin_id'] = df['bin_id'].astype(str)
    df['region'] = df['region'].astype(str)

    # Composite key: a bin_id that appears in multiple regions becomes distinct rows
    df['bin_key'] = df['bin_id'] + ' · ' + df['region']

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
                     region=('region', 'first')))

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
    for row in df.itertuples(index=False):
        items_array[
            bin_idx_map[row.bin_key],
            date_idx_map[row.date],
            pos_idx_map[row.position],
        ] = item_to_idx.get(str(row.item), -1)

    bin_meta = (
        df.drop_duplicates('bin_key')
        .set_index('bin_key')
        .loc[bin_keys, ['bin_rank', 'region']]
    )

    return {
        'items':       items_array,
        'bin_ranks':   bin_meta['bin_rank'].to_numpy().astype(int),
        'bin_regions': bin_meta['region'].to_numpy().astype(str),
        'bin_names':   np.array(bin_keys),
        'dates':       list(dates),
        'item_codes':  user_items,   # all items, frequency-ordered; no colours here
        'n_dup_keys':  n_dup_keys,
    }


# ============ HELPERS ============
def dim_color(hex_color, dim_amount=0.88, bg=PANEL_BG):
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

    counts = np.zeros((n_bins, n_positions, n_items), dtype=np.int32)
    for i in range(n_items):
        counts[:, :, i] = (items_subset == i).sum(axis=1)

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

    if 'regions' in p and 'regions_pills' not in st.session_state:
        val = [r for r in p['regions'].split(',') if r]
        if val:
            st.session_state['regions_pills'] = val

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


def make_view_csv(bin_names, positions, items_grid, share_grid, ranks, regions,
                  item_codes=None, bin_term='bin'):
    if item_codes is None:
        item_codes = ITEMS
    rows = []
    for j, bname in enumerate(bin_names):
        for i, pos in enumerate(positions):
            rows.append({
                bin_term:         bname,
                'rank':           int(ranks[j]),
                'region':         regions[j],
                'position':       int(pos),
                'item':           item_codes[int(items_grid[j, i])] if items_grid[j, i] >= 0 else '',
                'majority_share': round(float(share_grid[j, i]), 4),
            })
    return pd.DataFrame(rows).to_csv(index=False).encode()


# ============ APP ============
st.set_page_config(
    page_title="Ranked Placement Atlas",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom styling
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,400;0,500;1,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&display=swap" rel="stylesheet">
<style>
    .stApp {{ background-color: {BG}; }}
    .main .block-container {{ max-width: 1400px; padding-top: 2rem; }}
    .title-block {{
        border-bottom: 1px solid #2A2A2A;
        padding-bottom: 14px;
        margin-bottom: 18px;
    }}
    .eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: {MUTED};
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
        font-size: 10px !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        color: {MUTED} !important;
    }}
    div[data-baseweb="button-group"] button {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.04em !important;
    }}
    .stButton button {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 10px !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        border-radius: 0 !important;
        border: 1px solid #2A2A2A !important;
        background: transparent !important;
        color: {INK} !important;
        padding: 1px 8px !important;
        min-height: 26px !important;
        line-height: 1.4 !important;
    }}
    .stButton button:hover {{
        background: {INK} !important;
        color: {BG} !important;
    }}
    div[role="radiogroup"] label {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: {PANEL_BG};
    }}
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
data = generate_data()

with st.sidebar:
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:10px;'
        f'letter-spacing:0.18em;text-transform:uppercase;color:{MUTED};'
        f'margin-bottom:8px;">Data source</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Required columns: bin_id, date, position, item, bin_rank, region.",
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
            "Required columns: `bin_id`, `date`, `position`, `item`, `bin_rank`, `region`  \n"
            f"Up to {N_MAX_USER_ITEMS} items get distinct colours; extras shown in gray.  \n"
            "`bin_id` is used as the display name."
        )

    st.divider()
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:10px;'
        f'letter-spacing:0.18em;text-transform:uppercase;color:{MUTED};'
        f'margin-bottom:8px;">Labels</div>',
        unsafe_allow_html=True,
    )
    bin_term  = st.text_input("Bins are called", "bin",  key="bin_term").strip()  or "bin"
    item_term = st.text_input("Items are called", "item", key="item_term").strip() or "item"

    st.divider()
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:10px;'
        f'letter-spacing:0.18em;text-transform:uppercase;color:{MUTED};'
        f'margin-bottom:8px;">Share this view</div>',
        unsafe_allow_html=True,
    )
    components.html(
        f"""<script>
function copyAtlasLink(){{
  var btn=document.getElementById('share-btn');
  try{{
    navigator.clipboard.writeText(window.parent.location.href).then(function(){{
      btn.textContent='Copied!';
      setTimeout(function(){{btn.textContent='Copy link';}},2000);
    }});
  }}catch(e){{
    prompt('Copy this URL:',window.parent.location.href);
  }}
}}
</script>
<button id="share-btn" onclick="copyAtlasLink()"
  style="font-family:IBM Plex Mono,monospace;font-size:10px;letter-spacing:0.08em;
         text-transform:uppercase;border:1px solid #2A2A2A;background:transparent;
         padding:6px 14px;cursor:pointer;color:#1A1A1A;width:100%;">
  Copy link
</button>""",
        height=40,
    )
    st.caption("The link encodes your current filters, sort mode, and date range.")

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
    pill_items  = item_codes

# Reset items_pills when the pill options change (new file or new colour selection)
_item_sig = ','.join(pill_items)
if st.session_state.get('_item_sig') != _item_sig:
    st.session_state.pop('items_pills', None)
    st.session_state['_item_sig'] = _item_sig

available_regions = sorted(np.unique(data['bin_regions']).tolist())

# Reset regions_pills when the region vocabulary changes (e.g., new CSV uploaded)
_region_sig = ','.join(available_regions)
if st.session_state.get('_region_sig') != _region_sig:
    st.session_state.pop('regions_pills', None)
    st.session_state['_region_sig'] = _region_sig

# Per-item pill colors via JS — only pill_items get distinct colours.
# Discriminator: skip button groups that contain a region label.
_pill_colors  = [item_colors[item_codes.index(it)] for it in pill_items]
_colors_json  = json.dumps(_pill_colors)
_regions_json = json.dumps(available_regions)
components.html(
    f"""<script>
(function(){{
  var C={_colors_json}, BG="{BG}", N=C.length, R={_regions_json};
  function go(){{
    try{{
      var gs=window.parent.document.querySelectorAll('[data-baseweb="button-group"]');
      for(var i=0;i<gs.length;i++){{
        var bs=gs[i].querySelectorAll('button');
        if(bs.length!==N) continue;
        // Skip the regions group (contains known region labels)
        var texts=Array.from(bs).map(function(b){{return b.textContent.trim();}});
        if(R.some(function(r){{return texts.indexOf(r)>=0;}})) continue;
        for(var j=0;j<bs.length;j++){{
          var b=bs[j];
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
    new MutationObserver(go).observe(
      window.parent.document.body,
      {{subtree:true,childList:true,attributes:true,attributeFilter:['aria-pressed','aria-checked']}}
    );
  }}catch(e){{ setInterval(go,200); }}
}})();
</script>""",
    height=0,
)

# Apply URL query params to session state (only on fresh sessions)
apply_url_params(data['dates'], item_codes)

n_pos_total  = data['items'].shape[2]
min_rank_val = int(data['bin_ranks'].min())
max_rank_val = int(data['bin_ranks'].max())

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="title-block">
    <div class="title">Ranked Placement Atlas</div>
    <div class="subtitle">{len(data['bin_names'])} {bin_term}s × {n_pos_total} ranked positions × {n_items} {item_term}s × {len(data['dates'])} snapshots. When multiple dates are selected, each cell shows the modal {item_term} across the range (ties broken by recency).</div>
</div>
""", unsafe_allow_html=True)

# ── Row 1: Filters — Regions + Items ──────────────────────────────────────────
st.markdown(
    f'<style>.pill-btn button{{font-size:10px !important;padding:2px 10px !important;'
    f'height:auto !important;min-height:0 !important;}}</style>',
    unsafe_allow_html=True,
)

col_regions, col_items = st.columns([2, 5])

with col_regions:
    if 'regions_pills' in st.session_state:
        stored = st.session_state['regions_pills']
        valid = [r for r in stored if r in available_regions]
        if stored and not valid:
            del st.session_state['regions_pills']  # dataset changed, reset
        else:
            st.session_state['regions_pills'] = valid  # preserve empty (None clicked)

    _all_reg  = list(available_regions)
    def _reg_all():  st.session_state['regions_pills'] = _all_reg
    def _reg_none(): st.session_state['regions_pills'] = []

    selected_regions = st.pills(
        "Regions",
        available_regions,
        selection_mode="multi",
        default=available_regions,
        key="regions_pills",
    )
    rc1, rc2, _ = st.columns([2, 2, 4])
    with rc1:
        st.button("All",  key="btn_reg_all",  use_container_width=True, on_click=_reg_all)
    with rc2:
        st.button("None", key="btn_reg_none", use_container_width=True, on_click=_reg_none)

with col_items:
    if 'items_pills' not in st.session_state:
        st.session_state['items_pills'] = list(pill_items)

    _all_items = list(pill_items)
    def _items_all():  st.session_state['items_pills'] = _all_items
    def _items_none(): st.session_state['items_pills'] = []

    selected_items = st.pills(
        f"{item_term.capitalize()}s",
        pill_items,
        selection_mode="multi",
        key="items_pills",
    )
    st.caption(f"Click to highlight · unselected {item_term}s dim")
    ic1, ic2, _ = st.columns([1, 1, 8])
    with ic1:
        st.button("All",  key="btn_all",   on_click=_items_all)
    with ic2:
        st.button("None", key="btn_clear", on_click=_items_none)

# ── Row 2: Ranges — Date / Bin Rank / Position ────────────────────────────────
st.divider()

# Defaults for the one-rerun lag when the user switches date modes
step_idx  = 0
_n_window = len(data['dates'])
playing   = False

col_date, col_rank, col_pos = st.columns(3)

with col_date:
    _date_mode_pre = st.session_state.get('date_mode', 'Range')

    if _date_mode_pre == "Range":
        date_range = st.select_slider(
            "Date",
            options=data['dates'],
            value=(data['dates'][-13], data['dates'][-1]),
            format_func=lambda d: d.strftime("%b %d, '%y"),
            key="wk_slider",
        )
    else:
        # Dual-handle slider to define the stepping window
        _all_dates = data['dates']
        _sw_stored = st.session_state.get('step_range', (_all_dates[0], _all_dates[-1]))
        if _sw_stored[0] not in _all_dates or _sw_stored[1] not in _all_dates:
            _sw_stored = (_all_dates[0], _all_dates[-1])

        step_window = st.select_slider(
            "Date",
            options=_all_dates,
            value=_sw_stored,
            format_func=lambda d: d.strftime("%b %d, '%y"),
            key="step_range",
        )
        _range_start     = _all_dates.index(step_window[0])
        _range_end       = _all_dates.index(step_window[1])
        _dates_in_window = _all_dates[_range_start:_range_end + 1]
        _n_window        = len(_dates_in_window)

        _raw_idx = st.session_state.get('step_idx', 0)
        if _raw_idx >= _n_window:
            st.session_state['step_idx'] = max(0, _n_window - 1)
        step_idx  = st.session_state.get('step_idx', 0)
        step_date = _dates_in_window[step_idx]
        date_range = (step_date, step_date)

        st.caption(f"Step {step_idx + 1} of {_n_window}  ·  {step_date.strftime('%b %d, %Y')}")

        playing = st.session_state.get('playing', False)
        nb1, nb2, nb3, nb4 = st.columns(4)
        with nb1:
            if st.button("⏮", use_container_width=True, key="btn_first"):
                st.session_state.update({'step_idx': 0, 'playing': False})
                st.rerun()
        with nb2:
            if st.button("◀", use_container_width=True, key="btn_prev"):
                st.session_state.update({'step_idx': max(0, step_idx - 1), 'playing': False})
                st.rerun()
        with nb3:
            if st.button("⏸" if playing else "▶", use_container_width=True, key="btn_play"):
                st.session_state['playing'] = not playing
                st.rerun()
        with nb4:
            if st.button("▶▶", use_container_width=True, key="btn_next"):
                st.session_state.update({'step_idx': min(_n_window - 1, step_idx + 1), 'playing': False})
                st.rerun()

    date_mode = st.radio(
        "Mode", ["Range", "Step"],
        horizontal=True, key="date_mode",
    )
    if date_mode == "Range":
        st.session_state['playing'] = False

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

sort_mode = st.radio(
    f"Sort {bin_term}s by",
    sort_options,
    index=sort_options.index(st.session_state.get('sort_radio', 'Similarity')),
    horizontal=True,
    key="sort_radio",
)
st.caption(sort_descriptions(bin_term, item_term).get(sort_mode, ""))

# ── Filtering ─────────────────────────────────────────────────────────────────
regions_active = selected_regions if selected_regions else available_regions

in_region = np.isin(data['bin_regions'], regions_active)
in_rank   = (data['bin_ranks'] >= rank_range[0]) & (data['bin_ranks'] <= rank_range[1])
visible_mask        = in_region & in_rank
visible_bin_indices = np.where(visible_mask)[0]

date_start_idx = data['dates'].index(date_range[0])
date_end_idx   = data['dates'].index(date_range[1])
date_indices   = list(range(date_start_idx, date_end_idx + 1))

pos_indices = list(range(pos_range[0] - 1, pos_range[1]))

if len(visible_bin_indices) == 0 or len(date_indices) == 0 or len(pos_indices) == 0:
    st.warning("No data in current filter range. Widen your selectors.")
    st.stop()

# ── Majority computation ───────────────────────────────────────────────────────
items_sub = data['items'][visible_bin_indices][:, date_indices, :]
majority, share = compute_majority(items_sub, n_items)   # (n_vis, n_pos_total)

majority_f = majority[:, pos_indices]
share_f    = share[:, pos_indices]

# ── Sort ──────────────────────────────────────────────────────────────────────
# In Step mode, sort is anchored to the most recent date so rows stay stable
# as you step through time.
n_vis = len(visible_bin_indices)
if date_mode == "Step":
    _anchor = data['items'][visible_bin_indices][:, [-1], :]
    majority_sort, _ = compute_majority(_anchor, n_items)
else:
    majority_sort = majority

if sort_mode == "Index":
    order = np.arange(n_vis)
elif sort_mode == f"{bin_term.capitalize()} Rank":
    order = np.argsort(data['bin_ranks'][visible_bin_indices], kind='stable')
elif sort_mode == "Similarity":
    top4  = majority_sort[:, :4]
    keys  = [''.join(f'{int(x):x}' for x in row) for row in top4]
    order = np.argsort(keys, kind='stable')
elif sort_mode == "Top-rank":
    keys  = [''.join(f'{int(x):x}' for x in row) for row in majority_sort]
    order = np.argsort(keys, kind='stable')
elif sort_mode == "Selected Share":
    sel_idx_set = [item_codes.index(i) for i in selected_items]
    top10       = majority_sort[:, :10]
    share_count = np.isin(top10, sel_idx_set).sum(axis=1)
    order       = np.argsort(-share_count, kind='stable')
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
        return OTHER_COLOR   # gray items never dim further
    if all_or_none or i in sel_idx_set:
        return item_colors[i]
    return dim_color(item_colors[i], 0.88)

# Colorscale spans [-1, n_items]: slot 0 = no-data (background), slots 1..n_items = items.
_n = n_items + 1
colorscale = [[0.0, PANEL_BG], [1 / _n, PANEL_BG]]
for i in range(n_items):
    c = effective_color(i)
    colorscale.append([(i + 1) / _n, c])
    colorscale.append([(i + 2) / _n, c])

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
regions_disp   = data['bin_regions'][ordered_bin_indices]
bin_names_disp = data['bin_names'][ordered_bin_indices]

customdata = np.empty((n_show_bins, n_show_pos, 5), dtype=object)
customdata[:, :, 0] = bin_names_disp[:, None]
customdata[:, :, 1] = ranks_disp[:, None].astype(int)
customdata[:, :, 2] = regions_disp[:, None]
customdata[:, :, 3] = positions_disp[None, :].astype(int)
customdata[:, :, 4] = share_disp.astype(float)

_codes    = np.array(item_codes)
text_grid = np.where(
    majority_disp >= 0,
    _codes[np.clip(majority_disp.astype(int), 0, n_items - 1)],
    '—',
)
z         = majority_disp.astype(float)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_parts = []
for item in pill_items:
    idx  = item_codes.index(item)
    c    = effective_color(idx)
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
    '<div style="display:flex;flex-wrap:wrap;gap:2px;margin-bottom:6px;">'
    + ''.join(legend_parts) + '</div>',
    unsafe_allow_html=True,
)

# ── View summary ──────────────────────────────────────────────────────────────
date_count = len(date_indices)
multi_date = date_count > 1

if date_mode == "Step":
    _playing_badge = (
        ' · <span style="color:#B91C1C;font-weight:500;">▶ playing</span>'
        if st.session_state.get('playing') else ''
    )
    d0 = date_range[0].strftime("%b %d, %Y")
    _step_num = step_idx + 1
    mode_sentence = (
        f"Step mode — snapshot {_step_num} of {_n_window} ({d0}). "
        f"Row order is anchored to the most recent snapshot so bins stay in place as you step."
    )
    _date_label = f"snapshot {_step_num}/{_n_window}{_playing_badge}"
elif multi_date:
    d0 = date_range[0].strftime("%b %d, %Y")
    d1 = date_range[1].strftime("%b %d, %Y")
    mode_sentence = (
        f"Each cell shows the <b>most frequent {item_term}</b> across the {date_count} selected snapshots "
        f"({d0} → {d1}), with ties broken by the most recent week. "
        f"Hover to see the majority share — how often that {item_term} actually held the position."
    )
    _date_label = f"{date_count} snapshots"
else:
    d0 = date_range[0].strftime("%b %d, %Y")
    mode_sentence = f"Showing a single snapshot ({d0}): each cell is the {item_term} at that position."
    _date_label = "1 snapshot"

summary_html = f"""
<div style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;color:#4A4A4A;
            line-height:1.6;margin-bottom:10px;max-width:900px;
            border-left:2px solid #2A2A2A;padding-left:10px;">
    <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                letter-spacing:0.14em;text-transform:uppercase;color:{MUTED};
                margin-bottom:4px;">View summary</div>
    <b>{n_show_bins}</b> {bin_term}{'s' if n_show_bins != 1 else ''} ·
    <b>{n_show_pos}</b> position{'s' if n_show_pos != 1 else ''} ·
    {_date_label} ·
    regions: <b>{', '.join(sorted(set(regions_active)))}</b> ·
    sort: <b>{sort_mode}</b>
    <div style="margin-top:5px;color:#4A4A4A;">{mode_sentence}</div>
</div>
"""
st.markdown(summary_html, unsafe_allow_html=True)

# ── Cell size + Export row ────────────────────────────────────────────────────
_sz_col, _af_col, _gap_col, _dl_csv_col, _dl_html_col = st.columns([2, 1, 2, 1, 1])
with _sz_col:
    st.slider("Cell size (px)", 6, 28, cell_size, key="cell_sz")
with _af_col:
    st.write("")
    st.checkbox("Auto-fit", value=auto_fit, key="auto_fit_cb")

csv_bytes = make_view_csv(
    bin_names_disp, positions_disp, majority_disp, share_disp,
    ranks_disp, regions_disp, item_codes=item_codes, bin_term=bin_term,
)
with _dl_csv_col:
    st.download_button(
        "Download CSV",
        csv_bytes,
        file_name="atlas_view.csv",
        mime="text/csv",
        use_container_width=True,
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
            "<b>%{customdata[0]}</b>  ·  Rank %{customdata[1]}  ·  %{customdata[2]}<br>"
            "Position %{customdata[3]}  ·  Item <b>%{text}</b><br>"
            "Majority share: %{customdata[4]:.0%}"
            "<extra></extra>"
        ),
        xgap=0.5,
        ygap=0.5,
    ),
    row=1, col=1,
)

pos_dist = np.stack([(majority_disp == i).sum(axis=0) for i in range(n_items)], axis=1)

for item_idx in range(n_items):
    fig.add_trace(
        go.Bar(
            x=positions_disp,
            y=pos_dist[:, item_idx] / max(n_show_bins, 1),
            orientation='v',
            marker=dict(color=effective_color(item_idx), line=dict(width=0)),
            name=item_codes[item_idx],
            showlegend=False,
            hovertemplate=f"<b>{item_codes[item_idx]}</b>: %{{y:.0%}} at pos %{{x}}<extra></extra>",
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
    plot_bgcolor=PANEL_BG,
    paper_bgcolor=PANEL_BG,
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
    title=dict(text=f'Share by {bin_term}', font=dict(size=9, family='IBM Plex Mono', color=MUTED)),
    row=2, col=1,
)

# ── Render chart ──────────────────────────────────────────────────────────────
chart_event = st.plotly_chart(
    fig,
    use_container_width=False,
    on_select="rerun",
    key="main_chart",
)

# ── HTML export — pure Python, no subprocess, works everywhere ─────────────────
with _dl_html_col:
    _html = fig.to_html(include_plotlyjs='cdn', config={'displayModeBar': True})
    st.download_button(
        "Download HTML",
        _html.encode(),
        file_name="atlas_view.html",
        mime="text/html",
        use_container_width=True,
    )

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
        bin_region  = data['bin_regions'][bidx]
        bin_weekly  = data['items'][bidx]               # (n_weeks, n_pos_total)
        drill_items = bin_weekly[date_indices][:, pos_indices]  # (n_dates, n_pos)
        drill_dates = data['dates'][date_start_idx:date_end_idx + 1]

        _region_suffix = f" · {bin_region}" if ' · ' not in drill_bin else ""
        with st.expander(
            f"Time series · {drill_bin} · Rank {bin_rank_v}{_region_suffix}",
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
                    "Week: <b>%{x}</b>  ·  Position: <b>%{y}</b><br>"
                    "Item: <b>%{text}</b><extra></extra>"
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
                plot_bgcolor=PANEL_BG, paper_bgcolor=PANEL_BG,
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
            st.plotly_chart(mini_fig, use_container_width=False)

            # CSV for this bin's time series
            drill_rows = []
            for wi, d in enumerate(drill_dates):
                for pi, pos in enumerate(positions_disp):
                    drill_rows.append({
                        bin_term: drill_bin, 'rank': int(bin_rank_v), 'region': bin_region,
                        'date': d.isoformat(), 'position': int(pos),
                        'item': item_codes[int(drill_items[wi, pi])] if drill_items[wi, pi] >= 0 else '',
                    })
            drill_csv = pd.DataFrame(drill_rows).to_csv(index=False).encode()
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
    'regions': ','.join(sorted(set(regions_active))),
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

# ── Step-mode auto-play ────────────────────────────────────────────────────────
# Sleep AFTER the chart has been sent to the browser, then advance and rerun.
if date_mode == "Step" and st.session_state.get('playing', False):
    _step = st.session_state.get('step_idx', 0)
    _sw   = st.session_state.get('step_range', (data['dates'][0], data['dates'][-1]))
    _sw_s = data['dates'].index(_sw[0]) if _sw[0] in data['dates'] else 0
    _sw_e = data['dates'].index(_sw[1]) if _sw[1] in data['dates'] else len(data['dates']) - 1
    _n_win_auto = _sw_e - _sw_s + 1
    if _step < _n_win_auto - 1:
        time.sleep(0.5)
        st.session_state['step_idx'] = _step + 1
        st.rerun()
    else:
        st.session_state['playing'] = False
