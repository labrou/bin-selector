"""
Ranked Placement Atlas - Streamlit version
==========================================

Run:
    pip install streamlit>=1.40 plotly numpy pandas
    streamlit run app.py

Features:
  - Categorical heatmap: bins × positions × items × snapshots
  - Region / rank / position / date filters with shareable URL state
  - Sort modes: Index, Similarity, <bin_term> Rank, Top-rank, Selected Share
  - Aggregation methods: Majority, Abs. Majority, Weighted
  - Item highlight (dims non-selected items)
  - Legend mapping item codes to colors; VARIOUS shown as a distinct colour
  - Click any bin row to open its full time-series heatmap
  - Download current view as CSV or HTML
  - Upload your own CSV to replace the synthetic demo data

Data schema (uploaded CSV):
    Required columns: bin_id, date, position, item, bin_rank, segment
    Optional column:  N_item  (observation count for that item at that key;
                               defaults to 1 per row if absent)
    One row per unique [bin_id, date, position, segment, item].
    group_N (total observations per cell) is computed internally as
    sum(N_item) over all items sharing the same
    [bin_id, date, position, segment] key.
    bin_rank is bin-level metadata (same value for every row of a bin).
    Dates must be in M/D/YYYY format (single- or double-digit month/day).

Aggregation methods
-------------------
  Majority      (M1): per-date plurality winner (most observations on that date),
                      then cross-date majority count; tiebreak = most recent date.
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

from core import (
    N_MAX_ITEMS, N_MAX_USER_ITEMS, VARIOUS_LABEL, VARIOUS_COLOR,
    OTHER_LABEL, OTHER_COLOR, ITEMS, COLORS, SEGMENTS, BIN_NAMES,
    NUM_BINS, NUM_POSITIONS, NUM_DATES, METHOD_OPTIONS,
    BG, INK, MUTED, TITLE_FONTS,
    SORT_GUIDE_URL, METHOD_GUIDE_URL, VIZ_GUIDE_URL,
    generate_data, compute_plurality, compute_abs_majority,
    compute_weighted, compute_view, dim_color, make_view_csv, sort_descriptions,
    parse_uploaded_csv, compute_sort_order,
)


# ============ DATA GENERATION (Streamlit-cached wrappers) ============
@st.cache_resource          # returns by reference — no pickle/unpickle copy overhead
def _generate_data_cached():
    return generate_data()


@st.cache_data
def discover_items(file_bytes: bytes, filename: str):
    """Fast first pass: return item vocabulary ranked by total N_item.
    N_item is optional; defaults to 1 per row when absent."""
    try:
        # Read item column; include N_item only when it exists in the file
        cols = pd.read_csv(io.BytesIO(file_bytes), nrows=0).columns.tolist()
        read_cols = ['item', 'N_item'] if 'N_item' in cols else ['item']
        df = pd.read_csv(io.BytesIO(file_bytes), usecols=read_cols)
        df['item'] = df['item'].astype(str)
        if 'N_item' in df.columns:
            df['N_item'] = pd.to_numeric(df['N_item'], errors='coerce').fillna(1)
        else:
            df['N_item'] = 1
        vc = df.groupby('item')['N_item'].sum().sort_values(ascending=False)
        return vc.index.tolist(), vc.to_dict()
    except Exception:
        return [], {}


@st.cache_resource          # returns by reference — avoids copying large numpy arrays each rerun
def load_user_data(file_bytes: bytes, filename: str):
    """Parse a pre-aggregated CSV into compact date_winner/date_top_share arrays
    and sparse long arrays for the Weighted method.

    Delegates all pure data-transformation logic to parse_uploaded_csv in core.py
    and translates the returned messages into Streamlit calls.
    """
    data_dict, messages = parse_uploaded_csv(file_bytes)
    for level, text in messages:
        if level == "error":
            st.error(text)
        elif level == "warning":
            st.warning(text)
        elif level == "success":
            st.success(text)
    return data_dict


# ============ APP HELPERS ============

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
_title_font_key = st.session_state.get('title_font', 'IBM Plex Sans')
_title_font_css, _title_font_style = TITLE_FONTS.get(_title_font_key, TITLE_FONTS['IBM Plex Sans'])
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
    .js-plotly-plot .plotly .nsewdrag,
    .js-plotly-plot .plotly .ewdrag,
    .js-plotly-plot .plotly .nsdrag {{
        cursor: default !important;
    }}
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
data = _generate_data_cached()

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
            "Required columns: bin_id, date, position, item, bin_rank, segment. "
            "Optional: N_item (observation count; defaults to 1 if absent). "
            "One row per unique [bin_id, date, position, segment, item] combination. "
            "bin_rank is bin-level metadata, not part of the aggregation key."
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
            _candidate_colored = chosen if chosen else items_by_freq[:N_MAX_USER_ITEMS]
        else:
            _candidate_colored = items_by_freq

        user_data = load_user_data(uploaded.getvalue(), uploaded.name)
        if user_data is not None:
            data = user_data
            # Filter to items that survived cleaning; items only in dropped rows
            # (bad dates, bad positions, zero N_item) won't be in item_codes and
            # would crash item_codes.index(it) at the colour-assignment step.
            _valid_item_set = set(user_data['item_codes'])
            _filtered = [it for it in _candidate_colored if it in _valid_item_set]
            colored_items = _filtered if _filtered else user_data['item_codes'][:N_MAX_USER_ITEMS]
            _dw_any = data['date_winner'] if data['date_winner'] is not None else data['date_winner_by_filter'][0]
            n_b, n_d, n_p = _dw_any.shape
            n_it = len(data['item_codes'])
            st.success(
                f"{uploaded.name}\n\n"
                f"{n_b} bins · {n_p} positions · {n_d} snapshots · {n_it} items"
            )
    else:
        st.caption("Using synthetic demo data. Upload a pre-aggregated CSV to use your own data.")
        st.caption(
            "Required columns: `bin_id`, `date`, `position`, `item`, `bin_rank`, `segment`  \n"
            "Optional: `N_item` (observation count per row; defaults to 1 if absent).  \n"
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
    if data.get('filter_values'):
        filter_term = st.text_input("Filter attribute", "filter",
                                    key="filter_term").strip() or "filter"
    else:
        filter_term = st.session_state.get("filter_term", "filter") or "filter"

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

# ── Demo data banner ──────────────────────────────────────────────────────────
if uploaded is None:
    st.info("📊 Demo data — open the sidebar to upload your own CSV.", icon=None)

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
    st.session_state['items_pills'] = list(pill_items)
    st.session_state['_item_sig'] = _item_sig

available_filters = list(data.get('filter_values') or [])

_filter_sig = ','.join(available_filters)
if st.session_state.get('_filter_sig') != _filter_sig:
    st.session_state.pop('filter_pills', None)
    st.session_state['_filter_sig'] = _filter_sig

# Determine active filter now (before available_segments) so that segment
# pills and visible bins are restricted to only the bins that carry that
# filter value. Different provenances may have completely disjoint bin sets.
if available_filters and data.get('date_winner_by_filter') is not None:
    _pre_filter = st.session_state.get('filter_pills')
    if _pre_filter not in available_filters:
        _pre_filter = available_filters[0]
    _pre_fi = data['filter_values'].index(_pre_filter)
    _filter_bin_mask = np.any(data['date_winner_by_filter'][_pre_fi] >= 0, axis=(1, 2))
    available_segments = sorted(np.unique(data['bin_segments'][_filter_bin_mask]).tolist())
else:
    _pre_filter = None
    _pre_fi = None
    _filter_bin_mask = None
    available_segments = sorted(np.unique(data['bin_segments']).tolist())

_segment_sig = (_pre_filter or '') + '|' + ','.join(available_segments)
if st.session_state.get('_segment_sig') != _segment_sig:
    st.session_state['segments_pills'] = list(available_segments)
    st.session_state['_segment_sig'] = _segment_sig

# Reset date/rank/pos sliders when the dataset changes so stale values
# from a previous dataset (e.g. synthetic dates) don't crash list.index().
_dataset_sig = (
    f"{data.get('_id', 'synthetic')}"
    f"__{data['dates'][0].isoformat()}__{data['dates'][-1].isoformat()}"
    f"__{len(data['dates'])}"
    f"__{int(data['bin_ranks'].min())}__{int(data['bin_ranks'].max())}"
    f"__{(data['date_winner_by_filter'].shape[3] if data.get('date_winner_by_filter') is not None else data['date_winner'].shape[2])}"   # n_positions
)
if st.session_state.get('_dataset_sig') != _dataset_sig:
    for _k in ('wk_slider', 'rank_slider', 'pos_slider'):
        st.session_state.pop(_k, None)
    st.session_state['items_pills'] = list(pill_items)
    st.session_state['_dataset_sig'] = _dataset_sig

_items_sig = (_pre_filter or '') + '|' + ','.join(pill_items)
if st.session_state.get('_items_sig') != _items_sig:
    st.session_state['items_pills'] = list(pill_items)
    st.session_state['_items_sig'] = _items_sig

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
        var realBs=Array.from(gs[i].querySelectorAll('button'));
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

_dw_shape    = data['date_winner_by_filter'].shape[1:] if data.get('date_winner_by_filter') is not None else data['date_winner'].shape
n_pos_total  = _dw_shape[2]
min_rank_val = int(data['bin_ranks'].min())
max_rank_val = int(data['bin_ranks'].max())


# ── Dialogs ───────────────────────────────────────────────────────────────────
def _guide_subs():
    """Return substitution dict for %%KEY%% placeholders in guide files."""
    try:
        _host  = st.context.headers.get("host", "localhost:8502")
        _proto = "https" if not _host.startswith("localhost") else "http"
        _base  = f"{_proto}://{_host}"
    except Exception:
        _base = ""
    return dict(
        bin_term=bin_term,
        Bin_term=bin_term.capitalize(),
        item_term=item_term,
        Item_term=item_term.capitalize(),
        segment_term=segment_term,
        Segment_term=segment_term.capitalize(),
        filter_term=filter_term,
        Filter_term=filter_term.capitalize(),
        SORT_GUIDE_URL=f"{_base}/{SORT_GUIDE_URL}",
        METHOD_GUIDE_URL=f"{_base}/{METHOD_GUIDE_URL}",
        VIZ_GUIDE_URL=f"{_base}/{VIZ_GUIDE_URL}",
    )


def _load_guide_html(filename):
    """Load an HTML guide from static/, apply %%KEY%% substitutions, return rendered HTML."""
    text = (Path(__file__).parent / "static" / filename).read_text(encoding="utf-8")
    for key, val in _guide_subs().items():
        text = text.replace(f"%%{key}%%", str(val))
    style_m = re.search(r'<style>(.*?)</style>', text, re.DOTALL)
    body_m  = re.search(r'<body>(.*?)</body>',   text, re.DOTALL)
    if style_m and body_m:
        return f"<style>{style_m.group(1)}</style>{body_m.group(1)}"
    return text


def _load_guide_md(filename):
    """Load a Markdown guide from static/, apply %%KEY%% substitutions."""
    text = (Path(__file__).parent / "static" / filename).read_text(encoding="utf-8")
    for key, val in _guide_subs().items():
        text = text.replace(f"%%{key}%%", str(val))
    return text


@st.dialog("User Guide", width="large")
def _show_user_guide():
    try:
        st.markdown(_load_guide_md("user_guide.md"))
    except FileNotFoundError:
        st.markdown("User guide not found.")


@st.dialog("Sort modes — visual guide", width="large")
def _show_sort_guide():
    try:
        st.html(_load_guide_html("sort_modes_explainer.html"))
    except FileNotFoundError:
        st.markdown(f"[Open visual guide in browser →]({SORT_GUIDE_URL})")


@st.dialog("Method — visual guide", width="large")
def _show_method_guide():
    try:
        st.html(_load_guide_html("method_explainer.html"))
    except FileNotFoundError:
        st.markdown(f"[Open method guide in browser →]({METHOD_GUIDE_URL})")


@st.dialog("The visualization — visual guide", width="large")
def _show_viz_guide():
    try:
        st.html(_load_guide_html("visualization_explainer.html"))
    except FileNotFoundError:
        st.markdown(f"[Open visualization guide in browser →]({VIZ_GUIDE_URL})")


# ── Title ─────────────────────────────────────────────────────────────────────
_title_col, _viz_col, _help_col = st.columns([9, 1, 1])
with _title_col:
    n_bins_total  = _dw_shape[0]
    n_dates_total = _dw_shape[1]
    st.markdown(f"""
<div class="title-block">
    <div class="title" style="font-family:{_title_font_css};font-style:{_title_font_style};">{_custom_title}</div>
    <div class="subtitle">{n_bins_total} {bin_term}s &times; {n_pos_total} positions &times;
    {n_items} {item_term}s &times; {n_dates_total} snapshots.</div>
</div>
""", unsafe_allow_html=True)
with _viz_col:
    st.write("")
    st.write("")
    if st.button("User guide", key="help_btn"):
        _show_user_guide()
with _help_col:
    st.write("")
    st.write("")
    if st.button("Viz guide", key="viz_guide_btn"):
        _show_viz_guide()

# ── Row 1: Filter · Segments · Items · Method ─────────────────────────────────
# Change _FILTER_SELECTION_MODE to "multi" to allow multiple simultaneous selections.
_FILTER_SELECTION_MODE = "single"


def _reset_filters():
    for k in ('wk_slider', 'rank_slider', 'pos_slider', 'segments_pills', 'items_pills',
              '_dataset_sig', '_segment_sig', '_items_sig', '_item_sig'):
        st.session_state.pop(k, None)


# Row 1 header row with reset button
_row1_main, _row1_reset = st.columns([11, 1])
with _row1_reset:
    st.button("↺ Reset", key="btn_reset_filters", on_click=_reset_filters,
              help="Reset all filters to defaults")

if available_filters:
    if 'filter_pills' not in st.session_state or st.session_state['filter_pills'] not in available_filters:
        st.session_state['filter_pills'] = available_filters[0]
    col_filter, col_segments, col_items, col_method = st.columns(4)
else:
    selected_filter = None
    col_segments, col_items, col_method = st.columns(3)

if available_filters:
    with col_filter:
        _fhdr_col, _ = st.columns([4, 10], gap="small")
        with _fhdr_col:
            st.markdown(
                f'<p style="font-family:IBM Plex Mono,monospace;font-size:11px;'
                f'letter-spacing:0.15em;text-transform:uppercase;color:{INK};'
                f'margin:0;padding-top:5px;">{filter_term.capitalize()}</p>',
                unsafe_allow_html=True,
            )
        selected_filter = st.pills(
            filter_term.capitalize(),
            available_filters,
            selection_mode=_FILTER_SELECTION_MODE,
            key="filter_pills",
            label_visibility="collapsed",
        ) or available_filters[0]

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
    st.caption("Highlights only — does not filter rows.")

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
segments_active     = selected_segments   # empty list → no segment selected → zero bins
in_segment          = np.isin(data['bin_segments'], segments_active) if segments_active else np.zeros(len(data['bin_segments']), dtype=bool)
in_rank             = (data['bin_ranks'] >= rank_range[0]) & (data['bin_ranks'] <= rank_range[1])
visible_mask        = in_segment & in_rank
if _filter_bin_mask is not None:
    visible_mask   &= _filter_bin_mask
visible_bin_indices = np.where(visible_mask)[0]

date_start_idx = data['dates'].index(date_range[0])
date_end_idx   = data['dates'].index(date_range[1])
date_indices   = list(range(date_start_idx, date_end_idx + 1))

pos_indices = list(range(pos_range[0] - 1, pos_range[1]))

if len(visible_bin_indices) == 0 or len(date_indices) == 0 or len(pos_indices) == 0:
    st.warning("No data in current filter range. Widen your selectors.")
    st.stop()

# ── Compute view (session-state cache) ─────────────────────────────────────
# Resolve filter-specific dense arrays (None when no filter column is present).
if _pre_fi is not None:
    _active_fi  = _pre_fi
    _active_dw  = data['date_winner_by_filter'][_active_fi]
    _active_dts = data['date_top_share_by_filter'][_active_fi]
else:
    _active_fi  = None
    _active_dw  = data.get('date_winner')
    _active_dts = data.get('date_top_share')

# M1/M2: slices compact (B,D,P) arrays — sub-millisecond even without cache.
# M3: aggregates sparse long arrays; cache avoids re-filtering on sort/highlight.
_view_sig = (
    st.session_state.get('_dataset_sig', ''),
    tuple(visible_bin_indices.tolist()),
    date_start_idx, date_end_idx,
    tuple(pos_indices),
    method, n_items,
    selected_filter,
)
if st.session_state.get('_view_sig') != _view_sig:
    majority_view, share_view, weights_view = compute_view(
        data, visible_bin_indices, date_start_idx, date_end_idx,
        pos_indices, method, n_items, VARIOUS_IDX,
        filter_idx=_active_fi, date_winner_arr=_active_dw, date_top_share_arr=_active_dts,
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
n_vis = len(visible_bin_indices)
order = compute_sort_order(
    sort_mode, majority_view,
    data['bin_ranks'][visible_bin_indices],
    VARIOUS_IDX,
    selected_items or [], item_codes, pos_indices, n_pill_items,
)

ordered_bin_indices = visible_bin_indices[order]
majority_disp       = majority_view[order]
share_disp          = share_view[order]
weights_disp        = weights_view[order] if weights_view is not None else None

n_show_bins = len(ordered_bin_indices)
n_show_pos  = len(pos_indices)

# ── Colorscale ────────────────────────────────────────────────────────────────
sel_idx_set = set(item_codes.index(i) for i in selected_items) if selected_items else set()
all_or_none = len(sel_idx_set) >= n_pill_items  # empty = nothing highlighted (not "all")

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
cell_size = st.session_state.get('cell_sz', 28)
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
positions_disp = np.array(data['positions'])[pos_indices]
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
    bin_term, tuple(item_codes),
)
if st.session_state.get('_csv_sig') != _csv_sig:
    _new_csv = make_view_csv(
        bin_names_disp, positions_disp, majority_disp, share_disp,
        ranks_disp, segments_disp,
        item_codes=item_codes, bin_term=bin_term,
        method=method, weights_grid=weights_disp,
        colored_item_codes=pill_items,
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
        textfont=dict(size=max(7, min(int(cell_h) // 3, 11))),
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
    dragmode=False,
    clickmode='event+select',
)
fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, fixedrange=True, row=1, col=1)
fig.update_xaxes(
    showgrid=False, zeroline=False, fixedrange=True,
    tickvals=xtickvals, ticktext=[str(v) for v in xtickvals],
    tickfont=dict(size=9, family='IBM Plex Mono', color=MUTED),
    title=dict(text='POSITION', font=dict(size=9, family='IBM Plex Mono', color=MUTED)),
    row=2, col=1,
)
fig.update_yaxes(
    showgrid=False, zeroline=False, fixedrange=True,
    autorange='reversed',
    tickfont=dict(size=9, family='IBM Plex Mono', color=INK),
    row=1, col=1,
)
fig.update_yaxes(
    showgrid=False, zeroline=False, fixedrange=True, range=[0, 1],
    tickvals=[0, 0.5, 1], ticktext=['0%', '50%', '100%'],
    tickfont=dict(size=9, family='IBM Plex Mono', color=MUTED),
    title=dict(text=bar_y_label, font=dict(size=9, family='IBM Plex Mono', color=MUTED)),
    row=2, col=1,
)

# ── Render ─────────────────────────────────────────────────────────────────────
st.plotly_chart(
    fig, **_chart_own_width, key="main_chart",
    config={"modeBarButtonsToRemove": ["zoom2d","pan2d","zoomIn2d","zoomOut2d","autoScale2d","resetScale2d","lasso2d","select2d"], "displaylogo": False},
)

# ── HTML export key ───────────────────────────────────────────────────────────
_html_view_key = (
    st.session_state.get('_dataset_sig', ''),
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

# ── Chart tools — belong to the main heatmap above ────────────────────────────
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
        with st.spinner("Generating…"):
            st.session_state['_html_bytes'] = fig.to_html(
                include_plotlyjs='cdn', config={'displayModeBar': True}
            ).encode()
        st.rerun()

# ── Drill-down ────────────────────────────────────────────────────────────────
st.divider()
clicked_bin_name = None
drill_col, _ = st.columns([2, 3])
with drill_col:
    _no_sel      = f"— select a {bin_term} —"
    drill_options = [_no_sel] + list(bin_names_disp)
    default_idx   = 0
    if clicked_bin_name and clicked_bin_name in drill_options:
        default_idx = drill_options.index(clicked_bin_name)
    st.markdown(
        f'<p style="font-family:IBM Plex Sans,sans-serif;font-size:13px;'
        f'color:{MUTED};margin:0 0 2px;">Time series</p>',
        unsafe_allow_html=True,
    )
    drill_bin = st.selectbox(
        "Time series",
        drill_options,
        index=default_idx,
        key="drill_select",
        label_visibility="collapsed",
    )

if drill_bin != _no_sel:
    bin_matches = np.where(data['bin_names'] == drill_bin)[0]
    if len(bin_matches) > 0:
        bidx       = bin_matches[0]
        bin_rank_v = data['bin_ranks'][bidx]
        bin_seg    = data['bin_segments'][bidx]

        drill_dates = data['dates'][date_start_idx:date_end_idx + 1]

        # Per-date winner and top-item share from compact arrays (no 4D counts needed)
        drill_winner = _active_dw[bidx,
                           date_start_idx:date_end_idx + 1, :][:, pos_indices].copy()
        drill_share  = _active_dts[bidx,
                           date_start_idx:date_end_idx + 1, :][:, pos_indices].copy()
        # shape: (n_dates_sel, n_pos_sel)

        # For Abs. Majority: mark VARIOUS where top-item share < 50 %
        if method == 'Abs. Majority':
            drill_winner = np.where(
                (drill_winner >= 0) & (drill_share < 0.5),
                np.int32(VARIOUS_IDX),
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

            # Drill-down is always per-date regardless of method (single-date M3 == M1).
            # date_top_share = N_item[winner] / group_N for that specific date.
            if method == 'Abs. Majority':
                drill_share_label = "Plurality item pct"  # < 50 % for VARIOUS cells
            else:
                drill_share_label = "Date share"  # same value for Majority and Weighted
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
            mini_cell_h = max(9,  min(30, 600 // max(n_show_pos, 1)))
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
                autorange='reversed', showgrid=False, zeroline=False, fixedrange=True,
                tickfont=dict(size=9, family='IBM Plex Mono', color=INK),
                title=dict(text='POSITION', font=dict(size=9, family='IBM Plex Mono', color=MUTED)),
            )
            mini_fig.update_xaxes(
                showgrid=False, zeroline=False, fixedrange=True, tickangle=45,
                tickfont=dict(size=8, family='IBM Plex Mono', color=MUTED),
            )
            st.plotly_chart(mini_fig, **_chart_own_width,
                            config={"displayModeBar": False})

            # CSV for this bin's time series
            n_di, n_dp = drill_winner.shape
            drill_csv = pd.DataFrame({
                bin_term:   drill_bin,
                'rank':     int(bin_rank_v),
                'segment':  bin_seg,
                'date':     np.repeat([d.isoformat() for d in drill_dates], n_dp),
                'position': np.tile(positions_disp.astype(int), n_di),
                'item':     drill_text.ravel(),
                'date_share': np.round(drill_share.ravel(), 4),
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

