"""
Ranked Placement Atlas - Streamlit version
==========================================

Run:
    pip install streamlit>=1.40 plotly numpy
    streamlit run app.py

Data schema (synthetic):
    100 bins × 52 weekly snapshots × 50 ranked positions × 10 items (3-char codes).
    Each bin carries:
        - bin_rank: global rank 1-100 (ties allowed)
        - region:   one of NA, SA, EU, AS, CN, AU (constant per bin)
    Drift model:
        - Slow archetype drift across weeks
        - 20% of bins undergo a one-time regime change between weeks 15-37
"""

import json
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
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
NUM_BINS = 100
NUM_POSITIONS = 50
NUM_WEEKS = 52

BG = '#F7F4ED'
PANEL_BG = '#FBFAF5'
INK = '#1A1A1A'
MUTED = '#6B6B6B'

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

    # Bin-level attributes
    bin_archetypes = rng.integers(0, 5, NUM_BINS)
    bin_regions = np.array(rng.choice(REGIONS, NUM_BINS))

    # Bin rank: correlated with archetype + noise (ties expected)
    archetype_rank_center = np.array([15, 35, 50, 70, 50])
    rank_noise = (rng.random(NUM_BINS) + rng.random(NUM_BINS) + rng.random(NUM_BINS) - 1.5) * 18
    bin_ranks_raw = archetype_rank_center[bin_archetypes] + rank_noise
    bin_ranks = np.clip(np.round(bin_ranks_raw), 1, 100).astype(int)

    # Regime change for 20% of bins
    has_regime = rng.random(NUM_BINS) < 0.20
    regime_weeks = rng.integers(15, 38, NUM_BINS)
    regime_new_arch = rng.integers(0, 5, NUM_BINS)

    # Drift direction per bin
    drift_strength = 0.30
    drift_direction = rng.normal(0, 1, (NUM_BINS, 10))

    # Weekly dates ending today
    end_date = date.today()
    dates = [end_date - timedelta(weeks=NUM_WEEKS - 1 - w) for w in range(NUM_WEEKS)]

    items_array = np.zeros((NUM_BINS, NUM_WEEKS, NUM_POSITIONS), dtype=np.int8)

    for b in range(NUM_BINS):
        for w in range(NUM_WEEKS):
            if has_regime[b] and w >= regime_weeks[b]:
                base = archetypes[regime_new_arch[b]].copy()
            else:
                base = archetypes[bin_archetypes[b]].copy()

            # Apply smooth drift
            drift_factor = (w / (NUM_WEEKS - 1)) * drift_strength
            base = base + drift_direction[b] * drift_factor
            base = np.clip(base, 0.01, None)
            base = base / base.sum()

            # Vectorized sampling across all 50 positions
            combined = base[None, :] * pos_biases
            norm = combined / combined.sum(axis=1, keepdims=True)
            cumulative = np.cumsum(norm, axis=1)
            r = rng.random((NUM_POSITIONS, 1))
            picks = (r < cumulative).argmax(axis=1)
            items_array[b, w] = picks

    return {
        'items': items_array,            # (100, 52, 50)
        'bin_ranks': bin_ranks,          # (100,)
        'bin_regions': bin_regions,      # (100,)
        'bin_names': np.array(BIN_NAMES),  # (100,)
        'dates': dates,                  # list of 52 date objects
    }

# ============ HELPERS ============
def dim_color(hex_color, dim_amount=0.88, bg=PANEL_BG):
    """Blend hex_color toward background by dim_amount (0=no dim, 1=full bg)."""
    fg = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
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
    majority = counts.argmax(axis=2).astype(np.int8)

    # Tiebreak: prefer item from most recent week if it's tied for max
    most_recent = items_subset[:, -1, :]
    b_idx, p_idx = np.meshgrid(np.arange(n_bins), np.arange(n_positions), indexing='ij')
    most_recent_count = counts[b_idx, p_idx, most_recent]
    recent_is_max = most_recent_count == max_counts
    majority = np.where(recent_is_max, most_recent, majority).astype(np.int8)

    share = max_counts / n_weeks
    return majority, share

# ============ APP ============
st.set_page_config(page_title="Ranked Placement Atlas", layout="wide", initial_sidebar_state="collapsed")

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
    /* Streamlit widget label restyling */
    [data-testid="stWidgetLabel"] p {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 10px !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        color: {MUTED} !important;
    }}
    /* Pills */
    div[data-baseweb="button-group"] button {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.04em !important;
    }}
    /* Buttons */
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
    /* Radio horizontal */
    div[role="radiogroup"] label {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px !important;
    }}
</style>
""", unsafe_allow_html=True)

# Per-item pill colors via JS: CSS nth-child is unreliable because Streamlit's
# BaseWeb pills may wrap each button in its own div, making every button
# nth-child(1) of its wrapper. JS queries directly by button index instead.
# Discriminator: the items group has exactly len(ITEMS)=10 buttons; regions has 6.
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

st.markdown(f"""
<div class="title-block">
    <div class="eyebrow">Fig. 01 · Bin × Position × Item × Date</div>
    <div class="title">Ranked Placement Atlas</div>
    <div class="subtitle">100 bins × 50 ranked positions × 10 items × 52 weekly snapshots. When multiple dates are selected, each cell shows the modal item across the range (ties broken by recency).</div>
</div>
""", unsafe_allow_html=True)

data = generate_data()

# ===== CONTROL ROW 1: Regions + Items =====
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

# ===== CONTROL ROW 2: Date / Rank / Position ranges =====
col_date, col_rank, col_pos = st.columns(3)

with col_date:
    date_range = st.select_slider(
        "Date range",
        options=data['dates'],
        value=(data['dates'][-13], data['dates'][-1]),
        format_func=lambda d: d.strftime("%b %d, '%y"),
    )

with col_rank:
    rank_range = st.slider("Bin rank range", 1, 100, (1, 100))

with col_pos:
    pos_range = st.slider("Position range", 1, 50, (1, 50))

# ===== CONTROL ROW 3: Sort + Cell size =====
col_sort, col_size = st.columns([3, 2])

SORT_DESCRIPTIONS = {
    "Index":          "Original bin order — no analytical grouping; useful as a stable baseline.",
    "Similarity":     "Bins sharing the same items at positions 1–4 cluster together, surfacing archetypes as broad horizontal color bands. Default.",
    "Bin Rank":       "Top = highest-ranked bins (rank 1). Use this when the question is about rank: do top bins share a distinct profile?",
    "Top-rank":       "Full 50-position lexicographic sort. Position 1 dominates grouping, then position 2, etc. The leftmost column is perfectly grouped; later columns fragment.",
    "Selected Share": "Top = bins whose top-10 positions are most saturated by the selected items. Only available when 1–9 items are highlighted.",
}

n_sel = len(selected_items) if selected_items else 0
with col_sort:
    sort_options = ["Index", "Similarity", "Bin Rank", "Top-rank"]
    if 0 < n_sel < len(ITEMS):
        sort_options.append("Selected Share")
    sort_mode = st.radio(
        "Sort bins by",
        sort_options,
        index=1,
        horizontal=True,
    )
    st.caption(SORT_DESCRIPTIONS.get(sort_mode, ""))

with col_size:
    sc1, sc2 = st.columns([3, 2])
    with sc1:
        cell_size = st.slider("Cell size (px)", 6, 28, 12)
    with sc2:
        st.write("")
        auto_fit = st.checkbox("Auto-fit", value=True)

# ===== FILTERING =====
regions_active = selected_regions if selected_regions else REGIONS

in_region = np.isin(data['bin_regions'], regions_active)
in_rank = (data['bin_ranks'] >= rank_range[0]) & (data['bin_ranks'] <= rank_range[1])
visible_mask = in_region & in_rank
visible_bin_indices = np.where(visible_mask)[0]

date_start_idx = data['dates'].index(date_range[0])
date_end_idx = data['dates'].index(date_range[1])
date_indices = list(range(date_start_idx, date_end_idx + 1))

pos_indices = list(range(pos_range[0] - 1, pos_range[1]))

if len(visible_bin_indices) == 0 or len(date_indices) == 0 or len(pos_indices) == 0:
    st.warning("No data in current filter range. Widen your selectors.")
    st.stop()

# ===== MAJORITY COMPUTATION =====
items_sub = data['items'][visible_bin_indices][:, date_indices, :]
majority, share = compute_majority(items_sub)  # (n_vis, 50), (n_vis, 50)

# Apply position filter
majority_f = majority[:, pos_indices]
share_f = share[:, pos_indices]

# ===== SORT =====
n_vis = len(visible_bin_indices)
if sort_mode == "Index":
    order = np.arange(n_vis)
elif sort_mode == "Bin Rank":
    order = np.argsort(data['bin_ranks'][visible_bin_indices], kind='stable')
elif sort_mode == "Similarity":
    # Short fingerprint (top-4 positions) so bins sharing the same leading items
    # cluster into broad visible bands; ties resolved by stable sort.
    top4 = majority[:, :4]
    keys = [''.join(f'{int(x):x}' for x in row) for row in top4]
    order = np.argsort(keys, kind='stable')
elif sort_mode == "Top-rank":
    # Full 50-position lexicographic sort: fine-grained ordering within clusters.
    keys = [''.join(f'{int(x):x}' for x in row) for row in majority]
    order = np.argsort(keys, kind='stable')
elif sort_mode == "Selected Share":
    sel_idx_set = [ITEMS.index(i) for i in selected_items]
    top10 = majority[:, :10]
    share_count = np.isin(top10, sel_idx_set).sum(axis=1)
    order = np.argsort(-share_count, kind='stable')
else:
    order = np.arange(n_vis)

ordered_bin_indices = visible_bin_indices[order]
majority_disp = majority_f[order]
share_disp = share_f[order]

n_show_bins = len(ordered_bin_indices)
n_show_pos = len(pos_indices)

# ===== COLORSCALE =====
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

# ===== SIZING =====
# bins are rows (y), positions are columns (x)
container_w = 900   # plotting area width for positions
container_h = 720   # reference height for bins

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
total_width  = heatmap_width + 170      # left margin for bin name labels
total_height = int(heatmap_height / 0.83) + 60

# ===== BUILD FIGURE =====
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.85, 0.15],
    vertical_spacing=0.025,
    shared_xaxes=True,
)

positions_disp  = np.array(pos_indices) + 1  # 1-indexed for display
ranks_disp      = data['bin_ranks'][ordered_bin_indices]
regions_disp    = data['bin_regions'][ordered_bin_indices]
bin_names_disp  = data['bin_names'][ordered_bin_indices]

# Customdata for hover — shape (n_bins, n_positions, 5)
customdata = np.empty((n_show_bins, n_show_pos, 5), dtype=object)
customdata[:, :, 0] = bin_names_disp[:, None]
customdata[:, :, 1] = ranks_disp[:, None].astype(int)
customdata[:, :, 2] = regions_disp[:, None]
customdata[:, :, 3] = positions_disp[None, :].astype(int)
customdata[:, :, 4] = share_disp.astype(float)

text_grid = np.array(ITEMS)[majority_disp.astype(int)]

z = majority_disp.astype(float)  # rows=bins, cols=positions

# Main heatmap
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

# Bottom marginal: per-position item distribution across visible bins
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

xtickvals = sorted(set([int(positions_disp[0]), int(positions_disp[-1])] +
                       [p for p in positions_disp if int(p) % 5 == 0]))

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

# ===== VIEW SUMMARY =====
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
<div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 12px; color: #4A4A4A;
            line-height: 1.6; margin-bottom: 14px; max-width: 900px;
            border-left: 2px solid #2A2A2A; padding-left: 10px;">
    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 10px;
                letter-spacing: 0.14em; text-transform: uppercase; color: {MUTED};
                margin-bottom: 4px;">View summary</div>
    <b>{n_show_bins}</b> bin{'s' if n_show_bins != 1 else ''} ·
    <b>{n_show_pos}</b> position{'s' if n_show_pos != 1 else ''} ·
    <b>{date_count}</b> weekly snapshot{'s' if multi_date else ''} ·
    regions: <b>{', '.join(sorted(set(regions_active)))}</b> ·
    sort: <b>{sort_mode}</b>
    <div style="margin-top: 5px; color: #4A4A4A;">{mode_sentence}</div>
</div>
"""
st.markdown(summary_html, unsafe_allow_html=True)

st.plotly_chart(fig, use_container_width=False)
