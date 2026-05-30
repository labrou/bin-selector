# Ranked Placement Atlas

**Live app:** https://bin-selector.streamlit.app/

An interactive Streamlit visualization for exploring how a categorical
vocabulary (items) is distributed across ordered slots (positions) over many
groupings (bins), optionally aggregated across time.

The built-in synthetic dataset is **100 bins × 50 positions × 13 items × 52
snapshots**, but the app is designed to work with any CSV you upload.

---

## What can you visualize with this?

The atlas is domain-agnostic. Anything that fits the shape — *many ordered
containers, each filled with one of a set of categories, measured
repeatedly over time* — is a candidate. A few concrete examples:

| Your "bins" | Your "positions" | Your "items" | Question you can answer |
|---|---|---|---|
| **Retail stores / markets** | Shelf slots ranked 1–50 | Product SKUs or categories | Do top-ranked stores carry a distinct assortment in their primary slots? Which stores share a profile? |
| **Search result pages** | Organic rank 1–20 | Content type or brand | Which brands dominate position 1 across markets? Has the composition shifted over time? |
| **Streaming playlists** | Playlist position 1–30 | Genre or mood tag | Do high-engagement playlists cluster into archetypal shape patterns? |
| **Ad auction logs** | Ad slot rank 1–10 | Advertiser category | Which categories saturate the top slots? Does that vary by publisher region? |
| **Sports rosters** | Roster position 1–25 | Player role or stat tier | How do championship rosters differ from bottom-table ones at each slot? |
| **Feed / recommendation engines** | Feed slot 1–50 | Content category | How does position-1 content type vary across user cohorts or dates? |

Upload your own CSV (sidebar → Data source) to replace the synthetic demo
with real data. Use the **Labels** panel in the sidebar to rename "bin" and
"item" to whatever fits your domain.

---

## Contents

1. [Installation and running](#installation-and-running)
2. [Uploading your own data](#uploading-your-own-data)
3. [Data model](#data-model)
4. [The visualization](#the-visualization)
5. [Controls reference](#controls-reference)
6. [Sort modes](#sort-modes)
7. [Aggregation methods](#aggregation-methods)
8. [Code structure](#code-structure)
9. [Dependencies](#dependencies)

---

## Installation and running

```bash
pip install -r requirements.txt
streamlit run app.py
```

`requirements.txt`:

```
streamlit>=1.40
plotly>=5.18
numpy>=1.24
pandas>=1.5
```

`st.pills` and `st.dialog` require Streamlit 1.40 or newer.

---

## Uploading your own data

Use the **Data source** panel in the sidebar to upload a CSV. The app parses
it, keeps all items with their real names, and replaces the synthetic demo
for the session.

### CSV schema

| Column | Required | Notes |
|---|---|---|
| `bin_id` | ✓ | String identifier for the bin; used as the display name |
| `date` | ✓ | Format **M/D/YYYY** (month and day may be single-digit; year is 4 digits). Also accepts any format parseable by `pd.to_datetime`. |
| `position` | ✓ | Integer rank within the bin (1-based or 0-based; both work) |
| `item` | ✓ | Any string label. All unique items are retained and their real labels are always shown. Memory and computation scale with item count for the Weighted method (dense `B×P×items` array per view); datasets with hundreds of distinct items work well; thousands may be slow. |
| `bin_rank` | ✓ | Global rank of the bin (integer; any range, including 0-based) |
| `segment` | ✓ | Any string grouping / filter attribute; not restricted to a fixed set. Rename the display label via **Labels → Bin grouping attribute**. |
| `filter` | optional | Row-level provenance label. Use this to tag rows from different source files before concatenating them into one upload. All distinct values appear as pills; selecting one restricts the heatmap to rows that carry that label. Rename the display label via **Labels → Filter attribute**. |
| `N_item` | optional | Count of observations of this item for this `(bin_id, date, position, segment, item)` combination. **Defaults to 1 per row** when absent. |

`group_N` (total observations per cell across all items) is **computed internally** as `sum(N_item)` over all item rows sharing the same `(bin_id, date, position, segment)` key (plus `filter`, when the column is present). `bin_rank` is bin-level metadata (one value per bin) and is not part of the aggregation key. You do not need to include `group_N` in the file.

**Duplicate rows are aggregated automatically.** If the same `(bin_id, date, position, segment, item)` key (plus `filter`, when present) appears on multiple rows, their `N_item` values are summed before any processing. This means you can upload raw one-row-per-observation data (omitting `N_item`, so it defaults to 1) and the app will count correctly. Pre-aggregating to one row per key reduces file size but is not required.

There is no practical limit on the number of distinct items. All items are kept; only the top 10 by total frequency receive a distinct colour — the rest are shown in gray.

**Multiple segment values per bin_id.** If the same `bin_id` appears with
more than one `segment` value, each `(bin_id, segment)` pair is treated as a
distinct display unit. The heatmap label becomes `"bin_id · segment"`.

**Missing positions.** Bins do not need to have data for every position.
Cells where a bin has no data for a given position are shown as empty
(background color) with a `—` hover label.

### Item colours

All items are kept and their real names are always shown. Up to **10** items
can be given a distinct colour; everything beyond that renders in gray but
still displays its real label on hover and in cell text. When your data has
more than 10 unique item values, a multiselect in the sidebar lets you choose
which items get distinct colours (defaulting to the top 10 by frequency).

---

## Data model

### In-memory structure

Both `generate_data()` (synthetic) and `load_user_data()` (uploaded CSV)
return a dictionary with the following keys:

| Key               | Type         | Shape / size                    | Description |
|-------------------|--------------|---------------------------------|-------------|
| `date_winner`     | `np.int32`   | `(n_bins, n_dates, n_pos)`      | Per-date plurality winner item index for each cell. `-1` = no data. |
| `date_top_share`  | `np.float32` | `(n_bins, n_dates, n_pos)`      | Plurality winner's share of observations for that cell (0–1). |
| `wt_bin_idx`      | `np.int32`   | `(n_nonzero,)`                  | Bin index for each non-zero `(bin, date, pos, item)` observation (sparse long format). |
| `wt_date_idx`     | `np.int32`   | `(n_nonzero,)`                  | Date index — parallel to `wt_bin_idx`. |
| `wt_pos_idx`      | `np.int32`   | `(n_nonzero,)`                  | Position index — parallel to `wt_bin_idx`. |
| `wt_item_idx`     | `np.int32`   | `(n_nonzero,)`                  | Item index — parallel to `wt_bin_idx`. |
| `wt_N_item`       | `np.int32`   | `(n_nonzero,)`                  | Observation count for this `(bin, date, pos, item)` entry. |
| `wt_filter_idx`   | `np.int32` \| `None` | `(n_nonzero,)` or `None`  | Filter index — parallel to `wt_bin_idx`; `None` when the `filter` column is absent. |
| `date_winner_by_filter` | `np.int32` \| `None` | `(n_filters, n_bins, n_dates, n_pos)` or `None` | Per-filter-value plurality winner arrays; replaces `date_winner` when the `filter` column is present. |
| `date_top_share_by_filter` | `np.float32` \| `None` | `(n_filters, n_bins, n_dates, n_pos)` or `None` | Per-filter-value top-share arrays; replaces `date_top_share` when the `filter` column is present. |
| `filter_values`   | `list[str]` \| absent | `n_filters` entries or absent  | Sorted list of distinct filter labels (uploaded CSV only); absent in synthetic data. |
| `bin_ranks`       | `np.int64`   | `(n_bins,)`                     | Global rank per bin. |
| `bin_segments`    | `np.str_`    | `(n_bins,)`                     | Segment label per bin (the bin grouping attribute). |
| `bin_names`       | `np.str_`    | `(n_bins,)`                     | Display name per bin (used on y-axis). |
| `dates`           | `list[date]` | `n_dates` entries               | Date stamps from oldest to most recent. |
| `item_codes`      | `list[str]`  | `n_items` entries               | All item labels, frequency-ordered for uploaded data. |
| `item_colors`     | `list[str]`  | `n_items` entries               | Hex colour per item (distinct for top-10; gray for the rest). Synthetic only; not present in user data dict. |

**Design rationale.** `date_winner` / `date_top_share` are compact `(B, D, P)`
arrays — pre-computing the per-date plurality winner at load time means M1
and M2 aggregations require no per-item counting at interaction time, just
`np.bincount` over a flat int32 array. The sparse `wt_*` long arrays support
the Weighted aggregation (M3) without materialising a full 4D `(B, D, P, I)`
cube, which would be 15–40× larger in memory.

### Synthetic data generation

The built-in generator produces realistic structure for demonstrating the
visualization:

1. **Five archetypes.** Each bin is assigned to one of five item-probability
   distributions. Archetypes differ in which items dominate.
2. **Positional bias.** Positions 1–4 over-sample top items; positions 5–14
   over-sample mid items; positions 15+ sample uniformly.
3. **Rank correlation.** Archetype determines the bin's expected rank; Gaussian
   noise adds variance.
4. **Drift.** Each bin drifts in item-probability space over 52 weeks,
   producing gradual evolution.
5. **Regime change.** ~20% of bins undergo a one-time archetype switch between
   weeks 15 and 37, producing visible breaks in the time series.
6. **Multiple observations per cell.** Each `(bin, date, position)` cell is
   backed by 1–10 sampled observations, giving each of the three aggregation
   methods distinct behaviour. Some cells have clear majorities (>50%), others
   are contested; a few positions always have single-item dominance.

Reproducibility is guaranteed by fixed seed `42`.

---

## The visualization

### Main heatmap

A Plotly `Heatmap` trace where:

- **rows** are bins, labeled by name, ordered by the selected sort mode.
- **columns** are positions (position 1 at left).
- **cell color** encodes the item at that (bin, position) under the chosen
  [aggregation method](#aggregation-methods).
- When multiple dates are selected, each cell shows the aggregated item per
  the selected method (plurality winner, absolute majority, or weighted).
- **Hover** shows bin name, rank, region, position, item label, and the
  winning share or weight.

Items with a distinct colour are shown in that colour. Items beyond the colour
limit are shown in gray; their real label is still visible on hover and as the
cell text. Non-selected items (via the items pills) are dimmed by blending 88%
toward the background — except gray items, which are never dimmed further.

The cell labeled **VARIOUS** (Abs. Majority only) indicates that on the
majority of dates in the selected range no single item held ≥ 50% of that
day's observations — each such date voted VARIOUS, and VARIOUS won the
cross-date majority count.

### Bottom marginal

Stacked bar traces (one per item) sharing the x-axis with the heatmap. Each
bar shows the proportion of visible bins whose aggregated item at that position
is the given item. Recomputes from the current filter state on every
interaction.

---

## Controls reference

The controls are arranged above the chart (plus a cell-size / export row below the view summary).

### Row 1 — Filters

**Filter** (optional) · Shown only when the uploaded data contains a `filter` column.
Displays all distinct filter values as pills; exactly one value is active at a time
(no **all** / **none** buttons). Selecting a pill restricts the entire heatmap —
colours, shares, and bar chart — to rows whose `filter` value matches the selection.
The intended use case is concatenating multiple files of the same structure but different
provenance: add a `filter` column to tag each row's source, upload one combined file,
and switch between provenances using the pills. Absent when using the synthetic dataset
or when the CSV has no `filter` column. When present, this is the leftmost column of
Row 1. Rename the pill header via **Labels → Filter attribute**.

**`<region_term>`s** · Toggleable pills showing all values of the bin's grouping
attribute. Default: all selected. Filtering hides bins whose grouping value is
not selected. Rename this attribute via **Labels → Bin grouping attribute**.
**all** / **none** buttons below the pills select or clear all.

**Items** · Toggleable pills for the distinctly-coloured items (up to 10).
Gray items are always visible in the heatmap but are not shown as pills.
Selection controls *highlighting*, not filtering — non-selected items are
dimmed; gray items are unaffected. **all** / **none** buttons below the
pills reset or clear the selection.

**Method** · Three pills selecting the aggregation method applied when multiple
dates are in the date range. See [Aggregation methods](#aggregation-methods).

### Row 2 — Ranges

**Date** · Dual-handle `select_slider` over all dates in the data.
Default: most recent 13 snapshots. When the range covers a single date,
each cell is that date's value. For multiple dates, the
[aggregation method](#aggregation-methods) applies.

**Bin rank range** · Dual-handle slider over the actual rank range in the
data (min to max; 0-based ranks are supported). Bins outside the range are
hidden.

**Position range** · Dual-handle slider over `[1, n_positions]`. Positions
outside the range are hidden. Sort keys (Similarity, Top-rank) are computed
over the full position set, so row order is stable when the position window
changes.

### Row 3 — Sort

See [Sort modes](#sort-modes) below.

### Below heatmap — Cell size & Export

**Cell size** · Slider sets a preferred cell size in pixels (6–28 px,
default 12). **Auto-fit** expands cells to fill the available space:

```
cell_w = max(slider, container_width  / visible_positions)
cell_h = max(slider, container_height / visible_bins)
```

Cells are capped at 40 × 30 px.

**Download CSV** · Exports the current heatmap view (visible bins,
positions, and date range) as a CSV file.

**Download HTML** · Saves the interactive Plotly figure as a
self-contained HTML file.

All three controls appear in a single row directly below the heatmap.

### User guide

A **User guide** button in the top-right of the page opens a modal dialog
with a full walkthrough of every control.

### Sidebar

The sidebar (arrow at top-left) contains four sections:

**Data source** · Upload a CSV to replace the synthetic demo (see [Uploading your own data](#uploading-your-own-data)).

**Labels** · Two text inputs rename the domain vocabulary throughout the UI:

- **Bins are called** — replaces "bin" throughout the UI.
- **Items are called** — replaces "item" throughout the UI.
- **Filter attribute** — renames the filter pill header (visible only when the data has a `filter` column).

**Display** · Visual customisation:

- **Title** — editable heading text; defaults to "Ranked Placement Atlas".
- **Title font** — four options: *Fraunces* (italic serif, default), *Playfair Display*, *DM Serif Display*, *IBM Plex Sans*.
- **Background color** — color picker; updates the page, sidebar, chart backgrounds, and empty-cell color simultaneously.

**Share this view** · Displays the current URL (with all filters, sort, and date range encoded) for copying.

---

## Sort modes

All sort modes order the **rows** of the heatmap. Ties fall back to stable bin-index order.

**Visual guide with worked examples:** [sort_modes_explainer.html](https://labrou.github.io/bin-selector/sort_modes_explainer.html)

### Index

Bins in alphabetical order by bin ID — the unsorted baseline.

### Similarity

Bins sharing the same items at positions 1–4 cluster together. Each bin's
top-4 items are encoded as a hex string and sorted lexicographically.
Archetypes surface as broad horizontal color bands. **Default.**

### `<bin_term>` Rank

Ascending by `bin_rank` (highest-ranked bins at top). Use this to ask "do
the top-ranked bins share a distinct item profile?"

### Top-rank

Groups bins that share the same item at position 1; ties are resolved by
position 2, then 3, and so on — a strict left-to-right sort. Use this to find
bins with an identical opening sequence.

### Selected Share

Available when 1 to N−1 distinctly-coloured items are selected. Ranks bins
by how many of the **visible** positions are held by the selected items —
bins where your chosen items dominate rise to the top. Ties are broken by
which bin has the selected items at earlier (more prominent) positions.
Responds to the position range filter.

---

## Aggregation methods

When a date range spans multiple snapshots, each `(bin, position)` cell must
be resolved to a single item. Three methods are available, selectable via the
**Method** pills above the heatmap.

**Visual guide with worked examples:** [method_explainer.html](https://labrou.github.io/bin-selector/method_explainer.html)

### Majority (M1)

*The item that appears most often across the selected dates.*

For each `(bin, position)` cell, count appearances of each item across the
selected date range (dates where the cell has no data are skipped). Take the
item with the highest count. Ties are broken by the item that appears at the
most recent date. Each date contributes one vote regardless of the underlying
observation counts (`N_item`).

The hover tooltip shows **Date-win share** — winning count ÷ number of
dates with data for that cell.

### Abs. Majority (M2)

*Per-date threshold: a date's winner only counts if it held ≥ 50% of that day's observations — otherwise that date votes VARIOUS.*

Two-step logic:

1. **Per-date check.** For each date in the range, take the plurality winner and its share of observations on that date. If the share ≥ 50%, that date casts a vote for the item. If it falls short, that date votes **VARIOUS** instead.
2. **Cross-date majority.** Count votes across all dates (VARIOUS is a valid vote value, just like any item). The value with the most votes wins and is displayed.

A cell shows VARIOUS when the majority of dates had no single dominant item on that specific day — not just that no item won across the whole date range. Use this method when you want to surface cells with genuinely per-snapshot dominance and flag contested ones.

The hover tooltip shows **Date-win share** — the fraction of dates that voted for the displayed value (item or VARIOUS). For a VARIOUS cell the tooltip also notes *(no per-date majority)* to clarify that the displayed value is the contested outcome, not a real item.

VARIOUS cells are counted as their own "item" in the bottom marginal chart.

### Weighted (M3)

*Items weighted by observation counts, aggregated over the date range.*

Rather than giving each date a single vote, this method uses the raw
`N_item` counts from the uploaded data (or the synthetic sampling counts).
For each `(bin, position)` cell, the **weight** of an item equals:

```
weight(item) = sum(N_item for this item) / sum(N_item for all items)
               across all selected dates in the date range
```

The item with the highest total weight wins and is displayed. The hover
tooltip shows **Weighted share** (0–1) — the winning item's share of all
observations for that cell over the selected period.

Weighted is the only method sensitive to how many raw observations back each
date's entry — it surfaces items that appear rarely on any single date but
accumulate significant mass across the date range.

---

## Code structure

The codebase is organized into two files: `core.py` (compute logic and constants) and `app.py` (Streamlit UI and layout).

### `core.py` — Constants, data loading, and compute functions

**Constants:**
- Palette, colours, and display limits (N_MAX_ITEMS, COLORS, VARIOUS_COLOR, OTHER_COLOR)
- Synthetic data parameters (ITEMS, SEGMENTS, BIN_NAMES, NUM_BINS, NUM_POSITIONS, NUM_DATES)
- Guide file paths (SORT_GUIDE_URL, METHOD_GUIDE_URL, VIZ_GUIDE_URL)

**Data loading:**
- `generate_data()` — synthetic data generator with 5 archetypes, positional bias, drift, and regime changes; cached with fixed seed.
- `parse_uploaded_csv(file_bytes, filename)` — parses user CSV into compact `date_winner`/`date_top_share` arrays and sparse `wt_*` long arrays.

**Aggregation functions:**
- `compute_plurality(date_winner_slice, n_items)` — vectorised majority over dates via `np.bincount`.
- `compute_abs_majority(date_winner_slice, date_top_share_slice, n_items, various_idx)` — applies 50%-threshold per date.
- `compute_weighted(data, visible_bin_indices, date_start_idx, date_end_idx, pos_indices, n_items)` — aggregates sparse `wt_*` arrays; returns full 3D weights array for visualization.
- `compute_view(data, visible_bin_indices, date_start_idx, date_end_idx, pos_indices, method, n_items, various_idx)` — dispatcher that calls the appropriate aggregation function.

**Helpers:**
- `compute_sort_order(data, visible_bin_indices, pos_indices, mode)` — computes sort keys for all 5 sort modes.
- `dim_color(hex, amount, bg)` — blends a color toward the background.
- `make_view_csv(winner, share, visible_bins, pos_indices, bin_names, item_codes, dates)` — exports current view to CSV.
- `sort_descriptions(bin_term, item_term)` — returns sort-mode captions using domain vocabulary.

### `app.py` — Streamlit UI and caching

**Data generation section:**
- Streamlit `@st.cache_resource` and `@st.cache_data` wrappers around `generate_data()` and `parse_uploaded_csv()`.
- `discover_items(file_bytes, filename)` — fast first pass returning items ranked by frequency; cached.
- `load_user_data(file_bytes, filename)` — full CSV parse with progress updates; cached per file content hash.

**App helpers section:**
- `apply_url_params(dates, item_codes)` — seeds widget state from URL query params (enables shareable links).
- `_guide_subs()` — builds substitution dict for `%%KEY%%` placeholders in guide markdown files; constructs absolute URLs from host header.
- `_load_guide_html(filename)` — loads HTML guide, applies substitutions, renders via `st.markdown`.

**App layout section:**

Flat sequence of Streamlit calls executed top-to-bottom on every interaction:

1. Page config, CSS injection, theme setup.
2. Generate / load data; compute `_dataset_sig` for cache invalidation.
3. Sidebar: file upload → `discover_items` → colour-selection multiselect → `load_user_data` → domain terminology inputs.
4. Derive `item_colors` and `pill_items` from selected colours.
5. CSS injection for dynamic pill button colours.
6. Title block + **User guide** button (`@st.dialog`).
7. **Row 1 (Filters):** Filter pills (optional) + Segment/Region pills + Item pills + Method pills.
8. **Row 2 (Ranges):** Date range slider + Bin rank range slider + Position range slider.
9. **Row 3 (Sort):** Sort mode radio (5 options).
10. Session-state `_view_sig` cache check → `compute_view` (skips NumPy work on sort/highlight/cell-size changes).
11. Apply filters → compute sort order → build Plotly colorscale.
12. View summary block (bin count, position count, date range, method) + colour legend.
13. Heatmap + stacked marginal bar subplot (Plotly); responsive cell sizing.
14. Cell size slider + Auto-fit checkbox + Download CSV/HTML buttons.
15. Drill-down selectbox for per-bin time-series heatmap.

### Performance cache

Results from `compute_view` are stored in session state under the key
`_view_sig = (dataset_sig, visible_bins, date_start, date_end, pos_indices, method, n_items)`.
Interactions that change only sort order, item highlighting, or cell size
reuse the cached arrays and skip all NumPy work, keeping those interactions
near-instant.

### Colour assignment

```python
# colored_items: user's selection of up to 10 items
_color_idx  = {item: i for i, item in enumerate(colored_items)}
item_colors = [
    COLORS[_color_idx[item]] if item in colored_set else OTHER_COLOR
    for item in item_codes   # all items, frequency-ordered
]
```

The colorscale spans `[-1, n_items]`: slot 0 maps value `-1` (no data) to
the panel background; slots 1..n_items map each item index to its color
(distinct or gray); slot `n_items` maps `VARIOUS_IDX` to burnt-orange (`#C2410C`).

---

## Dependencies

| Package   | Minimum | Purpose |
|-----------|---------|---------|
| streamlit | 1.40    | Web app framework; `st.pills` and `st.dialog` require ≥ 1.40. |
| plotly    | 5.18    | Heatmap and subplot rendering. |
| numpy     | 1.24    | Vectorized array operations. |
| pandas    | 1.5     | CSV parsing and data manipulation. |

Optional fonts loaded from Google Fonts at runtime (no install):
Fraunces (display title), IBM Plex Mono (labels), IBM Plex Sans (body).
