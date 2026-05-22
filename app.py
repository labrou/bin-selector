"""
Ranked Placement Atlas - Streamlit version
==========================================

Run:
    pip install streamlit>=1.40 plotly numpy pandas kaleido
    streamlit run app.py

Features:
  - Categorical heatmap: bins × positions × items × weekly snapshots
  - Region / rank / position / date filters with shareable URL state
  - Sort modes: Index, Similarity, Bin Rank, Top-rank, Selected Share
  - Item highlight (dims non-selected items)
  - Legend mapping item codes to colors
  - Click any bin row to open its full time-series heatmap
  - Download current view as CSV or PNG
  - Upload your own CSV to replace the synthetic demo data

Data schema (synthetic):
    100 bins × 52 weekly snapshots × 50 ranked positions × 10 items (3-char codes).
    Each bin carries:
        - bin_rank: global rank 1-100 (ties allowed)
        - region:   one of NA, SA, EU, AS, CN, AU (constant per bin)
    Drift model:
        - Slow archetype drift across weeks
        - 20% of bins undergo a one-time regime change between weeks 15-37
"""

import io
import json
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

# ============ CONSTANTS ============
ITEMS = ['APX', 'BRT', 'CFD', 'DLT', 'ETR', 'FRM', 'GVS', 'HXC', 'INV', 'JTL']
COLORS = ['#B91C1C', '#1E3A8A', '#15803D', '#CA8A04', '#6D28D9',
          '#DB2777', '#0E7490', '#525252', '#92400E', '#4D7C0F']
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

SORT_DESCRIPTIONS = {
    "Index":          "Original bin order — no analytical grouping; useful as a stable baseline.",
    "Similarity":     "Bins sharing the same items at positions 1–4 cluster together, surfacing archetypes as broad horizontal color bands. Default.",
    "Bin Rank":       "Top = highest-ranked bins (rank 1). Use this when the question is about rank: do top bins share a distinct profile?",
    "Top-rank":       "Full 50-position lexicographic sort. Position 1 dominates grouping, then position 2, etc. The leftmost column is perfectly grouped; later columns fragment.",
    "Selected Share": "Top = bins whose top-10 positions are most saturated by the selected items. Only available when 1–9 items are highlighted.",
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
    bin_ranks       = np.clip(np.round(archetype_rank_center[bin_archetypes] + rank_noise), 1, 100).astype(int)

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
    items_array = (r < cumulative).argmax(axis=3).astype(np.int8)  # (B, W, P)

    return {
        'items':       items_array,
        'bin_ranks':   bin_ranks,
        'bin_regions': bin_regions,
        'bin_names':   np.array(BIN_NAMES),
        'dates':       dates,
    }


@st.cache_data
def load_user_data(file_bytes: bytes, filename: str):
    """Parse an uploaded CSV into the app data schema."""
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

    df['date'] = pd.to_datetime(df['date']).dt.date

    unknown_items = sorted(set(df['item'].unique()) - set(ITEMS))
    if unknown_items:
        st.error(f"Unknown item codes in CSV: {unknown_items}. Must be one of: {ITEMS}")
        return None

    unknown_regions = sorted(set(df['region'].unique()) - set(REGIONS))
    if unknown_regions:
        st.error(f"Unknown regions in CSV: {unknown_regions}. Must be one of: {REGIONS}")
        return None

    bin_ids  = sorted(df['bin_id'].unique())
    dates    = sorted(df['date'].unique())
    positions = sorted(df['position'].unique())

    n_bins  = len(bin_ids)
    n_weeks = len(dates)
    n_pos   = len(positions)

    item_to_idx  = {code: i for i, code in enumerate(ITEMS)}
    bin_idx_map  = {b: i for i, b in enumerate(bin_ids)}
    date_idx_map = {d: i for i, d in enumerate(dates)}
    pos_idx_map  = {p: i for i, p in enumerate(positions)}

    items_array = np.zeros((n_bins, n_weeks, n_pos), dtype=np.int8)
    for row in df.itertuples(index=False):
        items_array[
            bin_idx_map[row.bin_id],
            date_idx_map[row.date],
            pos_idx_map[row.position],
        ] = item_to_idx.get(row.item, 0)

    bin_meta = (
        df.drop_duplicates('bin_id')
        .set_index('bin_id')
        .loc[bin_ids, ['bin_rank', 'region']]
    )

    return {
        'items':       items_array,
        'bin_ranks':   bin_meta['bin_rank'].to_numpy().astype(int),
        'bin_regions': bin_meta['region'].to_numpy(),
        'bin_names':   np.array([str(b) for b in bin_ids]),
        'dates':       list(dates),
    }


# ============ HELPERS ============
def dim_color(hex_color, dim_amount=0.88, bg=PANEL_BG):
    """Blend hex_color toward background by dim_amount (0=no dim, 1=full bg)."""
    fg     = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    bg_rgb = tuple(int(bg[i:i+2], 16) for i in (1, 3, 5))
    blended = tuple(int(fg[i] * (1 - dim_amount) + bg_rgb[i] * dim_amount) for i in range(3))
    return f'#{blended[0]:02x}{blended[1]:02x}{blended[2]:02x}'


def compute_majority(items_subset):
    """
    items_subset shape: (n_bins, n_weeks, n_positions)
    Returns (majority shape (n_bins, n_positions), share shape (n_bins, n_positions))
    Pure mode with most-recent tiebreak.
    """
    n_bins, n_weeks, n_positions = items_subset.shape
    if n_weeks == 1:
        return items_subset[:, 0, :].astype(np.int8), np.ones((n_bins, n_positions))

    counts = np.zeros((n_bins, n_positions, 10), dtype=np.int32)
    for i in range(10):
        counts[:, :, i] = (items_subset == i).sum(axis=1)

    max_counts = counts.max(axis=2)
    majority   = counts.argmax(axis=2).astype(np.int8)

    most_recent       = items_subset[:, -1, :]
    b_idx, p_idx      = np.meshgrid(np.arange(n_bins), np.arange(n_positions), indexing='ij')
    most_recent_count = counts[b_idx, p_idx, most_recent]
    majority          = np.where(most_recent_count == max_counts, most_recent, majority).astype(np.int8)
    return majority, max_counts / n_weeks


def apply_url_params(dates):
    """On a fresh session, seed widget keys from URL query params."""
    p = st.query_params
    if not p:
        return

    if 'regions' in p and 'regions_pills' not in st.session_state:
        val = [r for r in p['regions'].split(',') if r in REGIONS]
        if val:
            st.session_state['regions_pills'] = val

    if 'items' in p and 'items_pills' not in st.session_state:
        st.session_state['items_pills'] = [i for i in p['items'].split(',') if i in ITEMS]

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
            st.session_state['rank_slider'] = (max(1, min(lo, hi)), min(100, max(lo, hi)))
        except ValueError:
            pass

    if 'ps0' in p and 'ps1' in p and 'pos_slider' not in st.session_state:
        try:
            lo, hi = int(p['ps0']), int(p['ps1'])
            n_pos  = len(dates)  # use dates length as proxy; actual cap applied by slider
            st.session_state['pos_slider'] = (max(1, min(lo, hi)), min(50, max(lo, hi)))
        except ValueError:
            pass

    if 'sort' in p and 'sort_radio' not in st.session_state:
        if p['sort'] in SORT_DESCRIPTIONS:
            st.session_state['sort_radio'] = p['sort']

    if 'cs' in p and 'cell_sz' not in st.session_state:
        try:
            st.session_state['cell_sz'] = max(6, min(28, int(p['cs'])))
        except ValueError:
            pass

    if 'af' in p and 'auto_fit_cb' not in st.session_state:
        st.session_state['auto_fit_cb'] = (p['af'] == '1')


def make_view_csv(bin_names, positions, items_grid, share_grid, ranks, regions):
    rows = []
    for j, bname in enumerate(bin_names):
        for i, pos in enumerate(positions):
            rows.append({
                'bin':            bname,
                'rank':           int(ranks[j]),
                'region':         regions[j],
                'position':       int(pos),
                'item':           ITEMS[int(items_grid[j, i])],
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

# Per-item pill colors via JS (MutationObserver; setInterval fallback for cross-origin).
# Discriminator: items group has exactly len(ITEMS)=10 buttons; regions group has 6.
_colors_json = json.dumps(COLORS)
components.html(
    f"""<script>
(function(){{
  var C={_colors_json}, BG="{BG}", N=C.length;
  function go(){{
    try{{
      var gs=window.parent.document.querySelectorAll('[data-baseweb="button-group"]');
      for(var i=0;i<gs.length;i++){{
        var bs=gs[i].querySelectorAll('button');
        if(bs.length!==N) continue;
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
        help="Columns: bin_id, date, position, item, bin_rank, region",
        label_visibility="collapsed",
    )
    if uploaded is not None:
        user_data = load_user_data(uploaded.getvalue(), uploaded.name)
        if user_data is not None:
            data = user_data
            n_b, n_w, n_p = data['items'].shape
            st.success(
                f"{uploaded.name}\n\n"
                f"{n_b} bins · {n_p} positions · {n_w} weeks"
            )
    else:
        st.caption("Using synthetic demo data. Upload a CSV to use your own data.")
        st.caption("Expected columns: `bin_id`, `date`, `position`, `item`, `bin_rank`, `region`")

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

# Apply URL query params to session state (only on fresh sessions)
apply_url_params(data['dates'])

n_pos_total  = data['items'].shape[2]
max_rank_val = int(data['bin_ranks'].max())

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="title-block">
    <div class="eyebrow">Fig. 01 · Bin × Position × Item × Date</div>
    <div class="title">Ranked Placement Atlas</div>
    <div class="subtitle">{len(data['bin_names'])} bins × {n_pos_total} ranked positions × {len(ITEMS)} items × {len(data['dates'])} weekly snapshots. When multiple dates are selected, each cell shows the modal item across the range (ties broken by recency).</div>
</div>
""", unsafe_allow_html=True)

# ── Control row 1: Regions + Items ────────────────────────────────────────────
col_regions, col_items = st.columns([2, 5])

with col_regions:
    selected_regions = st.pills(
        "Regions",
        REGIONS,
        selection_mode="multi",
        default=REGIONS,
        key="regions_pills",
    )

with col_items:
    if 'items_pills' not in st.session_state:
        st.session_state['items_pills'] = list(ITEMS)

    bc1, bc2, _ = st.columns([1, 1, 6])
    with bc1:
        if st.button("Select all", use_container_width=True, key="btn_all"):
            st.session_state['items_pills'] = list(ITEMS)
            st.rerun()
    with bc2:
        if st.button("Clear", use_container_width=True, key="btn_clear"):
            st.session_state['items_pills'] = []
            st.rerun()

    selected_items = st.pills(
        "Items (toggle to highlight; unselected items dim)",
        ITEMS,
        selection_mode="multi",
        key="items_pills",
    )

# ── Control row 2: Date / Rank / Position ─────────────────────────────────────
col_date, col_rank, col_pos = st.columns(3)

with col_date:
    date_range = st.select_slider(
        "Date range",
        options=data['dates'],
        value=(data['dates'][-13], data['dates'][-1]),
        format_func=lambda d: d.strftime("%b %d, '%y"),
        key="wk_slider",
    )

with col_rank:
    rank_range = st.slider(
        "Bin rank range", 1, max_rank_val, (1, max_rank_val),
        key="rank_slider",
    )

with col_pos:
    pos_range = st.slider(
        "Position range", 1, n_pos_total, (1, n_pos_total),
        key="pos_slider",
    )

# ── Control row 3: Sort + Cell size ───────────────────────────────────────────
col_sort, col_size = st.columns([3, 2])

n_sel = len(selected_items) if selected_items else 0
with col_sort:
    sort_options = ["Index", "Similarity", "Bin Rank", "Top-rank"]
    if 0 < n_sel < len(ITEMS):
        sort_options.append("Selected Share")

    # If stored sort is no longer valid (e.g. Selected Share removed), reset
    if st.session_state.get('sort_radio') not in sort_options:
        st.session_state['sort_radio'] = 'Similarity'

    sort_mode = st.radio(
        "Sort bins by",
        sort_options,
        index=sort_options.index(st.session_state.get('sort_radio', 'Similarity')),
        horizontal=True,
        key="sort_radio",
    )
    st.caption(SORT_DESCRIPTIONS.get(sort_mode, ""))

with col_size:
    sc1, sc2 = st.columns([3, 2])
    with sc1:
        cell_size = st.slider("Cell size (px)", 6, 28, 12, key="cell_sz")
    with sc2:
        st.write("")
        auto_fit = st.checkbox("Auto-fit", value=True, key="auto_fit_cb")

# ── Filtering ─────────────────────────────────────────────────────────────────
regions_active = selected_regions if selected_regions else REGIONS

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
majority, share = compute_majority(items_sub)   # (n_vis, n_pos_total)

majority_f = majority[:, pos_indices]
share_f    = share[:, pos_indices]

# ── Sort ──────────────────────────────────────────────────────────────────────
n_vis = len(visible_bin_indices)
if sort_mode == "Index":
    order = np.arange(n_vis)
elif sort_mode == "Bin Rank":
    order = np.argsort(data['bin_ranks'][visible_bin_indices], kind='stable')
elif sort_mode == "Similarity":
    top4  = majority[:, :4]
    keys  = [''.join(f'{int(x):x}' for x in row) for row in top4]
    order = np.argsort(keys, kind='stable')
elif sort_mode == "Top-rank":
    keys  = [''.join(f'{int(x):x}' for x in row) for row in majority]
    order = np.argsort(keys, kind='stable')
elif sort_mode == "Selected Share":
    sel_idx_set = [ITEMS.index(i) for i in selected_items]
    top10       = majority[:, :10]
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
sel_idx_set = set(ITEMS.index(i) for i in selected_items) if selected_items else set()
all_or_none = (len(sel_idx_set) == 0) or (len(sel_idx_set) == len(ITEMS))

def effective_color(i):
    if all_or_none or i in sel_idx_set:
        return COLORS[i]
    return dim_color(COLORS[i], 0.88)

colorscale = []
for i in range(10):
    c = effective_color(i)
    colorscale.append([i / 10, c])
    colorscale.append([(i + 1) / 10, c])

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

text_grid = np.array(ITEMS)[majority_disp.astype(int)]
z         = majority_disp.astype(float)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_parts = []
for idx, (item, color) in enumerate(zip(ITEMS, COLORS)):
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
st.markdown(
    '<div style="display:flex;flex-wrap:wrap;gap:2px;margin-bottom:6px;">'
    + ''.join(legend_parts) + '</div>',
    unsafe_allow_html=True,
)

# ── View summary ──────────────────────────────────────────────────────────────
date_count = len(date_indices)
multi_date = date_count > 1

if multi_date:
    d0 = date_range[0].strftime("%b %d, %Y")
    d1 = date_range[1].strftime("%b %d, %Y")
    mode_sentence = (
        f"Each cell shows the <b>most frequent item</b> across the {date_count} selected snapshots "
        f"({d0} → {d1}), with ties broken by the most recent week. "
        f"Hover to see the majority share — how often that item actually held the position."
    )
else:
    d0 = date_range[0].strftime("%b %d, %Y")
    mode_sentence = f"Showing a single snapshot ({d0}): each cell is the item at that position for that week."

summary_html = f"""
<div style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;color:#4A4A4A;
            line-height:1.6;margin-bottom:10px;max-width:900px;
            border-left:2px solid #2A2A2A;padding-left:10px;">
    <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                letter-spacing:0.14em;text-transform:uppercase;color:{MUTED};
                margin-bottom:4px;">View summary</div>
    <b>{n_show_bins}</b> bin{'s' if n_show_bins != 1 else ''} ·
    <b>{n_show_pos}</b> position{'s' if n_show_pos != 1 else ''} ·
    <b>{date_count}</b> weekly snapshot{'s' if multi_date else ''} ·
    regions: <b>{', '.join(sorted(set(regions_active)))}</b> ·
    sort: <b>{sort_mode}</b>
    <div style="margin-top:5px;color:#4A4A4A;">{mode_sentence}</div>
</div>
"""
st.markdown(summary_html, unsafe_allow_html=True)

# ── Export buttons ────────────────────────────────────────────────────────────
exp_c1, exp_c2, _ = st.columns([1, 1, 5])

csv_bytes = make_view_csv(
    bin_names_disp, positions_disp, majority_disp, share_disp,
    ranks_disp, regions_disp,
)
with exp_c1:
    st.download_button(
        "Download CSV",
        csv_bytes,
        file_name="atlas_view.csv",
        mime="text/csv",
        use_container_width=True,
    )

with exp_c2:
    try:
        png_bytes = None
        # Build figure first (done below); placeholder here — PNG built after fig
        _png_placeholder = st.empty()
    except Exception:
        pass

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
        zmin=0,
        zmax=10,
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

pos_dist = np.stack([(majority_disp == i).sum(axis=0) for i in range(10)], axis=1)

for item_idx in range(10):
    fig.add_trace(
        go.Bar(
            x=positions_disp,
            y=pos_dist[:, item_idx] / max(n_show_bins, 1),
            orientation='v',
            marker=dict(color=effective_color(item_idx), line=dict(width=0)),
            name=ITEMS[item_idx],
            showlegend=False,
            hovertemplate=f"<b>{ITEMS[item_idx]}</b>: %{{y:.0%}} at pos %{{x}}<extra></extra>",
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
    title=dict(text='Share by bin', font=dict(size=9, family='IBM Plex Mono', color=MUTED)),
    row=2, col=1,
)

# PNG export (requires kaleido)
with exp_c2:
    try:
        png_bytes = fig.to_image(format="png", width=total_width, height=total_height, scale=2)
        st.download_button(
            "Download PNG",
            png_bytes,
            file_name="atlas_view.png",
            mime="image/png",
            use_container_width=True,
        )
    except Exception:
        st.caption("`pip install kaleido` for PNG export")

# ── Render chart ──────────────────────────────────────────────────────────────
chart_event = st.plotly_chart(
    fig,
    use_container_width=False,
    on_select="rerun",
    key="main_chart",
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
    drill_options = ["— select a bin —"] + list(bin_names_disp)
    default_idx   = 0
    if clicked_bin_name and clicked_bin_name in drill_options:
        default_idx = drill_options.index(clicked_bin_name)
    drill_bin = st.selectbox(
        "Drill down — bin time series",
        drill_options,
        index=default_idx,
        key="drill_select",
    )

if drill_bin != "— select a bin —":
    bin_matches = np.where(data['bin_names'] == drill_bin)[0]
    if len(bin_matches) > 0:
        bidx        = bin_matches[0]
        bin_rank_v  = data['bin_ranks'][bidx]
        bin_region  = data['bin_regions'][bidx]
        bin_weekly  = data['items'][bidx]               # (n_weeks, n_pos_total)
        drill_items = bin_weekly[date_indices][:, pos_indices]  # (n_dates, n_pos)
        drill_dates = [data['dates'][i] for i in date_indices]

        with st.expander(
            f"Time series · {drill_bin} · Rank {bin_rank_v} · {bin_region}",
            expanded=True,
        ):
            mini_z    = drill_items.T.astype(float)           # (n_pos, n_dates)
            mini_text = np.array(ITEMS)[drill_items.T.astype(int)]
            x_labels  = [d.strftime("%b %d") for d in drill_dates]
            y_labels  = list(positions_disp)

            mini_fig = go.Figure(go.Heatmap(
                z=mini_z,
                x=x_labels,
                y=y_labels,
                colorscale=colorscale,
                zmin=0, zmax=10,
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
                        'bin': drill_bin, 'rank': int(bin_rank_v), 'region': bin_region,
                        'date': d.isoformat(), 'position': int(pos),
                        'item': ITEMS[int(drill_items[wi, pi])],
                    })
            drill_csv = pd.DataFrame(drill_rows).to_csv(index=False).encode()
            st.download_button(
                f"Download {drill_bin} time series CSV",
                drill_csv,
                file_name=f"{drill_bin.lower()}_timeseries.csv",
                mime="text/csv",
            )

# ── Update URL query params (reflects current state for sharing) ───────────────
st.query_params.update({
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
})
