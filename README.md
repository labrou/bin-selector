# Ranked Placement Atlas

**Live app:** https://bin-selector.streamlit.app/

An interactive Streamlit visualization for exploring how a categorical
vocabulary (items) is distributed across ordered slots (positions) over many
groupings (bins), optionally aggregated across time.

The built-in synthetic dataset is **100 bins × 50 positions × 10 items × 52
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
7. [Majority metric](#majority-metric)
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

The `st.pills` widget requires Streamlit 1.40 or newer.

---

## Uploading your own data

Use the **Data source** panel in the sidebar to upload a CSV. The app parses
it, keeps all items with their real names, and replaces the synthetic demo
for the session.

### CSV schema

| Column | Required | Notes |
|---|---|---|
| `bin_id` | ✓ | String identifier for the bin; used as the display name |
| `date` | ✓ | Any format parseable by `pd.to_datetime`; daily, weekly, or any frequency |
| `position` | ✓ | Integer rank within the bin (1-based or 0-based; both work) |
| `item` | ✓ | Any string label — no limit on unique values |
| `bin_rank` | ✓ | Global rank of the bin (integer; any range, including 0-based) |
| `region` | ✓ | Any string grouping label; not restricted to a fixed set |

**Multiple regions per bin_id.** If the same `bin_id` appears in more than one
region, each `(bin_id, region)` pair is treated as a distinct display unit.
The heatmap label becomes `"bin_id · region"`.

**Duplicate rows.** Multiple rows for the same `(bin_id, region, date,
position)` key are treated as repeated measurements. The most frequent item
wins; ties are broken by random choice. A sidebar warning shows how many keys
were affected.

**Missing positions.** Bins do not need to have data for every position. Cells
where a bin has no data for a given position are shown as empty (background
color) with a `—` hover label.

### Item colours

All items are kept and their real names are always shown. Up to **11** items
can be given a distinct colour; everything beyond that renders in gray but
still displays its real label on hover and in cell text. When your data has
more than 11 unique item values, a multiselect in the sidebar lets you choose
which items get distinct colours (defaulting to the top 11 by frequency).

---

## Data model

### In-memory structure

Both `generate_data()` (synthetic) and `load_user_data()` (uploaded CSV)
return a dictionary with the following keys:

| Key           | Type          | Shape                    | Description |
|---------------|---------------|--------------------------|-------------|
| `items`       | `np.int8`     | (n_bins, n_dates, n_pos) | Item index at each (bin, date, position). `-1` means no data for that cell. |
| `bin_ranks`   | `np.int64`    | (n_bins,)                | Global rank per bin. |
| `bin_regions` | `np.str_`     | (n_bins,)                | Region label per bin. |
| `bin_names`   | `np.str_`     | (n_bins,)                | Display name per bin (used on y-axis). |
| `dates`       | `list[date]`  | n_dates entries          | Date stamps from oldest to most recent. |
| `item_codes`  | `list[str]`   | n_items entries          | All item labels, frequency-ordered for uploaded data. |

Item colours are not stored in the data dictionary — they are computed in the
app from the user's colour-selection multiselect.

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

Reproducibility is guaranteed by fixed seed `42`.

---

## The visualization

### Main heatmap

A Plotly `Heatmap` trace where:

- **rows** are bins, labeled by name, ordered by the selected sort mode.
- **columns** are positions (position 1 at left).
- **cell color** encodes the item at that (bin, position) — or empty
  (background) if the bin has no data for that position.
- When multiple dates are selected, each cell shows the **modal item** across
  those dates, with ties broken by recency.
- **Hover** shows bin name, rank, region, position, item label, and majority
  share.

Items with a distinct colour are shown in that colour. Items beyond the colour
limit are shown in gray; their real label is still visible on hover and as the
cell text. Non-selected items (via the items pills) are dimmed by blending 88%
toward the background — except gray items, which are never dimmed further.

### Bottom marginal

Stacked bar traces (one per item) sharing the x-axis with the heatmap. Each
bar shows the proportion of visible bins whose majority item at that position
is the given item. Recomputes from the current filter state on every
interaction.

---

## Controls reference

### Regions

Toggleable pills showing all region values present in the data (dynamic —
not limited to a fixed set). Default: all selected. Filtering by region
hides bins whose region is not selected.

### Items

Toggleable pills showing only the **distinctly-coloured** items (up to 11).
Gray items are always visible in the heatmap but not shown as pills.

- **Select all** selects all pill items.
- **Clear** deselects all.

Item selection controls highlighting, not filtering. Non-selected items are
dimmed; gray items are unaffected by the dim.

### Date range

A `select_slider` over all dates in the data. Default: most recent 13 dates.
When the range covers a single date, each cell is that date's value. For
multiple dates, the [majority metric](#majority-metric) applies.

### Bin rank range

Dual-handle slider over the actual rank range in the data (min to max;
0-based ranks are supported). Bins outside the range are hidden.

### Position range

Dual-handle slider over `[1, n_positions]`. Positions outside the range are
hidden. Sort keys (Similarity, Top-rank) are computed over the full position
set, so row order is stable when the position window changes.

### Cell size and auto-fit

A slider sets a preferred cell size in pixels (6–28 px, default 12). **Auto-fit**
expands cells to fill the available space using:

```
cell_w = max(slider, container_width  / visible_positions)
cell_h = max(slider, container_height / visible_bins)
```

Cells are capped at 40 × 30 px. With auto-fit off, the slider value is used
as-is and the figure may scroll if many bins are visible.

### Labels

Two text inputs in the sidebar let you rename the domain vocabulary:

- **Bins are called** — replaces "bin" throughout the UI (e.g., "store",
  "artist", "cohort").
- **Items are called** — replaces "item" throughout the UI (e.g., "brand",
  "genre", "role").

---

## Sort modes

All sort modes order the **rows** of the heatmap. Keys are computed over
the full majority array (all positions), so narrowing the position range
does not change row order. Ties fall back to stable bin-index order.

### Index

Bins in their original data order — the unsorted baseline.

### Similarity

Bins sharing the same items at positions 1–4 cluster together. Each bin's
top-4 items are encoded as a hex string and sorted lexicographically.
Archetypes surface as broad horizontal color bands. **Default.**

### `<bin_term>` Rank

Ascending by `bin_rank` (highest-ranked bins at top). Use this to ask "do
the top-ranked bins share a distinct item profile?"

### Top-rank

Lexicographic sort over all positions. Position 1 is perfectly grouped; later
positions are sorted to resolve ties in earlier ones.

### Selected Share

Available when 1 to N−1 distinctly-coloured items are selected. Ranks bins
by how many of their top-10 positions are held by the selected items. Use this
to find bins that lean hardest on a specific item set.

---

## Majority metric

When multiple dates are selected, `compute_majority(items_subset, n_items)`
returns:

1. For each `(bin, position)` cell, count occurrences of each item across
   the selected dates (cells with no data, value `-1`, are ignored).
2. Take the item with the maximum count.
3. Tiebreak: prefer the item that appeared in the **most recent** date.
4. If no item ever appeared (all dates have no data for that cell), return
   `-1` (rendered as empty).

Each cell also produces a *majority share* — the winning item's count divided
by the number of selected dates — surfaced in the hover tooltip.

---

## Code structure

The app lives in a single file `app.py`, organized into five sections.

### Section 1: Constants

Fixed palette, synthetic bin names, and limits:

```python
N_MAX_ITEMS      = 12   # total colour slots including gray
N_MAX_USER_ITEMS = 11   # max user-selectable distinctly-coloured items
OTHER_COLOR      = '#9CA3AF'  # gray for items beyond the colour limit
COLORS           = ['#B91C1C', '#1E3A8A', ...]  # 12 qualitative colours
```

### Section 2: Data loading

- `discover_items(file_bytes, filename)` — fast first pass reading only the
  `item` column; returns items ordered by descending frequency and a count
  dict. Cached by file content.
- `load_user_data(file_bytes, filename)` — full parse; keeps all items with
  real names; handles composite `(bin_id, region)` keys and deduplication.
  Returns the data dictionary without colour information (colours are
  assigned in the UI).
- `generate_data()` — synthetic data generator. Cached with fixed seed.

### Section 3: Helpers

- `dim_color(hex, amount, bg)` — blends a color toward the background.
- `compute_majority(items_subset, n_items)` — vectorized mode + tiebreak.
- `apply_url_params(dates, item_codes)` — seeds widget state from URL query
  params on fresh sessions (enables shareable links).
- `make_view_csv(...)` — serialises the current heatmap view to CSV for
  download.
- `sort_descriptions(bin_term, item_term)` — returns sort-mode caption
  strings using the current domain vocabulary.

### Section 4: App layout

Flat sequence of widget calls and data transforms executed top-to-bottom
on every interaction:

1. Page config and CSS injection.
2. Generate synthetic data (default).
3. Sidebar: file upload → `discover_items` → colour-selection multiselect
   → `load_user_data` → terminology inputs.
4. Derive `item_colors` from the colour-selection; define `pill_items`.
5. JS injection for pill button colours.
6. **Control row 1:** Regions pills and Items pills.
7. **Control row 2:** Date, rank, and position sliders.
8. **Control row 3:** Sort mode radio and cell size controls.
9. Filter pipeline → `compute_majority` → sort → build colorscale.
10. Heatmap + marginal bar subplot → `st.plotly_chart`.
11. Drill-down selectbox for per-bin time-series heatmap.
12. Export buttons (CSV, HTML).

### Colour assignment

```python
# colored_items: user's selection of up to 11 items
_color_idx  = {item: i for i, item in enumerate(colored_items)}
item_colors = [
    COLORS[_color_idx[item]] if item in colored_set else OTHER_COLOR
    for item in item_codes   # all items, frequency-ordered
]
```

The colorscale spans `[-1, n_items]`: slot 0 maps value `-1` (no data) to
the panel background; slots 1..n_items map each item index to its color
(distinct or gray).

---

## Dependencies

| Package   | Minimum | Purpose |
|-----------|---------|---------|
| streamlit | 1.40    | Web app framework; `st.pills` requires ≥ 1.40. |
| plotly    | 5.18    | Heatmap and subplot rendering. |
| numpy     | 1.24    | Vectorized array operations. |
| pandas    | 1.5     | CSV parsing and data manipulation. |

Optional fonts loaded from Google Fonts at runtime (no install):
Fraunces (display title), IBM Plex Mono (labels), IBM Plex Sans (body).
