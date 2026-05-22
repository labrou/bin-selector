# Ranked Placement Atlas

**Live app:** https://bin-selector.streamlit.app/

An interactive Streamlit visualization for exploring how a small categorical
vocabulary (items) is distributed across a large set of ordered slots
(positions) over many groupings (bins), with optional aggregation over time.

The default synthetic dataset is **100 bins × 50 positions × 10 items × 52
weekly snapshots**, but the app generalizes to any data with that shape.

---

## What can you visualize with this?

The atlas is domain-agnostic. Anything that fits the shape — *many ordered
containers, each filled with one of a small set of categories, measured
repeatedly over time* — is a candidate. A few concrete examples:

| Your "bins" | Your "positions" | Your "items" | Question you can answer |
|---|---|---|---|
| **Retail stores / markets** | Shelf slots ranked 1–50 | Product SKUs or categories | Do top-ranked stores carry a distinct assortment in their primary slots? Which stores share a profile? |
| **Search result pages** | Organic rank 1–20 | Content type or brand | Which brands dominate position 1 across markets? Has the composition shifted week over week? |
| **Streaming playlists** | Playlist position 1–30 | Genre or mood tag | Do high-engagement playlists cluster into archetypal shape patterns? |
| **Ad auction logs** | Ad slot rank 1–10 | Advertiser category | Which categories saturate the top slots? Does that vary by publisher region? |
| **Sports rosters** | Roster position 1–25 | Player role or stat tier | How do championship rosters differ from bottom-table ones at each roster slot? |
| **Feed / recommendation engines** | Feed slot 1–50 | Content category | How does position-1 content type vary across user cohorts or dates? |

Upload your own CSV (sidebar → Data source) to replace the synthetic demo
with real data. Use the **Labels** panel in the sidebar to rename "bin" and
"item" to whatever fits your domain.

---

## Contents

1. [Overview](#overview)
2. [Installation and running](#installation-and-running)
3. [Data model](#data-model)
4. [The visualization](#the-visualization)
5. [Controls reference](#controls-reference)
6. [Sort modes](#sort-modes)
7. [Majority metric](#majority-metric)
8. [Code structure](#code-structure)
9. [Extending for real data](#extending-for-real-data)
10. [Dependencies](#dependencies)

---

## Overview

Each *bin* (e.g., a storefront, a venue, a slot owner) holds an ordered list of
50 *positions* ranked 1 to 50. Each position is filled with one of 10 *items*,
each represented by a 3-character code. The same bin can hold different items
at the same position on different *dates*. Bins additionally carry a global
*rank* (1 = top, 100 = bottom, ties allowed) and a *region* attribute (one of
six: NA, SA, EU, AS, CN, AU) that is constant per bin.

The atlas renders this four-dimensional structure as a categorical heatmap:

- Rows are bins, each labeled by name, ordered by a user-selected sort mode.
- Columns are positions (position 1 at left).
- Cell color encodes the item at that (bin, position).
- When multiple dates are selected, each cell shows the modal (most frequent)
  item across those dates, with ties broken by the most recent week.
- A bottom marginal shows the per-position item distribution across the
  currently visible bins.

Filters let the user narrow the view by region, rank range, position range,
and date range; selected items can be visually highlighted (others dim).

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
```

The `st.pills` widget used for multi-select toggles requires Streamlit 1.40 or
newer.

---

## Data model

### Schema

The in-memory data structure returned by `generate_data()` is a dictionary
with four keys:

| Key           | Type             | Shape          | Description                                     |
| ------------- | ---------------- | -------------- | ----------------------------------------------- |
| `items`       | `np.int8`        | (100, 52, 50)  | Item index 0–9 at each (bin, week, position).   |
| `bin_ranks`   | `np.int64`       | (100,)         | Global rank 1–100 per bin (ties allowed).       |
| `bin_regions` | `np.str_`        | (100,)         | One of NA, SA, EU, AS, CN, AU per bin.          |
| `bin_names`   | `np.str_`        | (100,)         | Display name for each bin (shown on y-axis).    |
| `dates`       | `list[date]`     | 52 entries     | Weekly date stamps from oldest to most recent.  |

For real data, you only need to populate these five arrays in the same shapes
and dtypes; the rest of the app is dataset-agnostic.

### Synthetic data generation

The synthetic generator in `generate_data()` produces realistic structure
suitable for demonstrating the visualization:

1. **Five archetypes.** Each bin is assigned to one of five archetype
   probability distributions over the 10 items. Archetype 0 skews heavily to
   APX/BRT; archetype 3 skews to mid-tail items; etc.
2. **Positional bias.** Positions 1–4 over-sample top items relative to the
   archetype; positions 5–14 over-sample mid items; positions 15+ sample
   uniformly from the archetype. This produces the characteristic "top
   positions look different from bottom positions" pattern.
3. **Rank correlation.** Bin rank centers depend on archetype (archetype 0
   averages rank 15, archetype 3 averages rank 70), with Gaussian noise.
   Regions are sampled uniformly and independently of rank.
4. **Drift.** Each bin has a random drift direction in 10-dimensional
   probability space; over the 52 weeks, the archetype's probabilities shift
   by 30% of the drift vector. This produces gradual evolution.
5. **Regime change.** About 20% of bins undergo a one-time archetype switch
   at a random week between 15 and 37, producing visible breaks in the time
   series for those bins.

Reproducibility is guaranteed by the fixed seed `42` passed to
`np.random.default_rng`.

---

## The visualization

### Main heatmap

The grid is rendered as a Plotly `Heatmap` trace with these properties:

- **z** is a 2D array of integer item indices (0–9), shape (bins, positions).
  Rows are bins; columns are positions.
- **y** is the array of bin names; **x** is the array of position numbers.
- **colorscale** is constructed dynamically: a discrete 10-step scale where
  each integer maps to a specific item color. When some but not all items
  are selected, non-selected items are pre-blended toward the background to
  produce a faded appearance — this is done at the color level (not via
  Plotly opacity) so the resulting figure stays a single trace.
- **customdata** carries per-cell metadata (bin name, rank, region, position,
  majority share) for the hover tooltip.
- **text** carries the 3-character item code, shown in hover.
- `xgap=0.5, ygap=0.5` adds a thin separator between cells.

### Bottom marginal

Ten stacked `Bar` traces (one per item, all `orientation='v'`) sharing the
x-axis with the heatmap. Each bar shows the proportion of currently visible
bins whose majority item at that position is the given item. The marginal
recomputes from the displayed data on every rerun, so the percentages always
reflect the current filter state.

---

## Controls reference

### Regions

Six toggleable pills (NA, SA, EU, AS, CN, AU). Default: all six selected. A
bin is included in the view only if its `bin_regions` value is in the
selected set. With no regions selected, the filter is treated as a no-op
(all regions visible) to avoid an empty view.

### Items

Ten toggleable pills (one per item code). Two helper buttons:

- **Select all** sets all 10.
- **Clear** unselects all.

Item selection controls highlighting, not filtering. Non-selected items
still appear in the heatmap but are dimmed by blending the cell color 88%
toward the background. With either 0 or all 10 items selected, the
highlighting is a no-op (full color throughout).

### Date range

A `select_slider` over the 52 weekly dates. The default selection is the
most recent 13 weeks. When the start and end of the range coincide on a
single date, the cell value is simply that date's item; otherwise, the
[majority metric](#majority-metric) applies.

### Bin rank range

A dual-handle slider over `[1, 100]`. Bins whose `bin_rank` falls outside
the selected range are filtered out.

### Position range

A dual-handle slider over `[1, 50]`. Positions outside the range are
hidden from the heatmap. This affects only what's *displayed* — sort modes
that use top-10 or top-50 sequences (Similarity, Top-rank) compute against
the full position set, so the row order remains stable when the position
window changes.

### Cell size and auto-fit

A slider sets a preferred cell size in pixels (range 6–28, default 12). The
**Auto-fit** checkbox, when on, computes the effective cell size as:

```
cell_w_effective = max(slider_value, container_width  / visible_positions)
cell_h_effective = max(slider_value, container_height / visible_bins)
```

This means cells expand to fill available space when the filter narrows the
visible set, and the slider acts as a floor / upsizer. With auto-fit off,
the slider value is used as-is and the figure may scroll vertically if many
bins are visible. Cells are capped at 40 px wide × 30 px tall to avoid a
degenerate large-button appearance with very small visible sets.

---

## Sort modes

All sort modes order the **bins** (rows) of the heatmap. Sort keys are
computed against the full majority array (across all 50 positions), so
narrowing the position range does not perturb row order. Ties are broken
by stable sort (preserving the underlying bin index order).

### Index

Bins appear in their original index order (bin 1, bin 2, …, bin 100), with
filtered-out bins simply omitted. This is the "no sort" baseline. The
resulting view typically looks unstructured because the bin index is not
analytically meaningful. Use it to confirm what the raw, unsorted data looks
like, or as a stable reference when comparing the effect of other sorts.

### Similarity

Bins sharing the same items at positions 1–4 cluster together as adjacent
rows. The implementation encodes each bin's top-4 item indices as a single
hexadecimal string (e.g., a bin whose items at positions 1–4 are
`[0, 0, 1, 2]` becomes `"0012"`) and sorts the resulting strings
lexicographically. Bins sharing position-1 item cluster first; within that
cluster, position-2 item determines sub-grouping; and so on. Archetypes
surface as broad horizontal color bands (groups of consecutive rows with
similar color patterns). This is the default sort.

A 4-position fingerprint is deliberately short: with 10 possible items per
position, a 10-character key would nearly uniquely identify every bin among
100, producing an order indistinguishable from Top-rank. Four positions
leaves enough ties to form visible clusters. To change the window, modify
`top4 = majority[:, :4]` in the sort block.

### Bin Rank

Bins are ordered by their external `bin_rank` value ascending, so the
top-most rows are the highest-ranked bins (rank 1, 2, 3, …) and the
bottom-most rows are the lowest-ranked. Bins with identical ranks fall
back to bin-index order under the stable sort. This sort is the natural
choice when the analytical question concerns rank: "do the top-ranked bins
share an item profile that mid-ranked or bottom-ranked bins don't?"
Combined with the bin-rank range slider, it allows isolating a band of
ranks (e.g., 1–25) and viewing them in priority order.

### Top-rank

Lexicographic sort over all 50 positions. The implementation is identical
to Similarity but uses the full majority array rather than the top-10
slice. Bins with the same item at position 1 group together; within those,
ties at position 1 are resolved by position 2; ties at position 2 by
position 3; and so on through position 50.

Visually, the leftmost column (position 1) is perfectly grouped — all bins
with APX at position 1 are bunched at the top, then all bins with BRT at
position 1, etc. The cost is that right-column structure is disrupted to
serve the left-column ordering. Use Top-rank when you specifically care
about distribution at position 1–3 and are willing to sacrifice cluster
legibility in later positions.

### Selected Share

This mode is available only when between 1 and 9 items are selected (at 0
or 10 selected items it is meaningless and the radio option is hidden).

For each bin, the sort key is the count of how many of its top-10 positions
are occupied by any of the selected items. Bins are sorted by this count
descending, so the top-most rows are the heaviest users of the selected
items in their featured positions. Ties are resolved by bin index.

Use Selected Share to answer questions like "which 20 bins lean hardest on
items APX, BRT, and CFD in their featured positions?" Combined with
filtering, you can quickly identify outlier bins by item mix within any
slice of the data.

To use a different position window for the share calculation (e.g., top 5
or top 20 instead of top 10), modify `top10 = majority[:, :10]` in the
`elif sort_mode == "Selected Share":` block. Note: this is a separate
slice from the Similarity fingerprint (`top4 = majority[:, :4]`).

---

## Majority metric

When more than one date is selected, each `(bin, position)` cell shows the
mode (most frequent item) across the selected date range. The metric is
"pure mode with most-recent tiebreak":

1. For each cell, count occurrences of each of the 10 items across the
   selected weeks.
2. Take the item with the maximum count as the majority.
3. If multiple items tie for the maximum, prefer the item that appeared in
   the most recent week within the selection. If the most-recent-week
   value is not among the tied items, fall back to the lowest item index
   (the default `argmax` behavior).

The implementation is vectorized in `compute_majority()`:

```python
counts = np.zeros((n_bins, n_positions, 10), dtype=np.int32)
for i in range(10):
    counts[:, :, i] = (items_subset == i).sum(axis=1)

max_counts = counts.max(axis=2)
majority = counts.argmax(axis=2)

# Tiebreak: most recent week
most_recent = items_subset[:, -1, :]
most_recent_count = counts[np.arange(n_bins)[:, None],
                            np.arange(n_positions)[None, :],
                            most_recent]
majority = np.where(most_recent_count == max_counts, most_recent, majority)
```

Each cell also computes a *majority share* — the winning item's count
divided by the number of selected weeks. This value is surfaced in the
hover tooltip (`"Majority share: 67%"`) and is available in `share_disp` if
you want to encode confidence visually (e.g., cell opacity tied to share).

When only one date is selected, the function short-circuits: each cell
simply takes that date's value, and the majority share is 1.0 by definition.

---

## Code structure

The entire app lives in a single file, `app.py`, organized into five
sections.

### Section 1: Constants

```python
ITEMS     = ['APX', 'BRT', 'CFD', 'DLT', 'ETR', 'FRM', 'GVS', 'HXC', 'INV', 'JTL']
COLORS    = ['#B91C1C', '#1E3A8A', ...]  # one per item
REGIONS   = ['NA', 'SA', 'EU', 'AS', 'CN', 'AU']
BIN_NAMES = ['Apex', 'Basin', 'Birch', ...]  # 100 short nouns, one per bin
NUM_BINS, NUM_POSITIONS, NUM_WEEKS = 100, 50, 52
```

Colors are a hand-picked qualitative palette designed for the cream
background (`#F7F4ED`). The set follows the recommendations for categorical
heatmaps with around 10 categories: maximally distinct hues, balanced
saturation and value, no two colors near-isoluminant when viewed together.

### Section 2: Data generation

`generate_data()` runs once and is cached with `@st.cache_data`. See the
[data model](#data-model) section for what it produces and how. To swap in
real data, replace the body of this function with code that loads your
data and returns a dictionary with the four documented keys.

### Section 3: Helpers

Two pure functions:

- `dim_color(hex_color, dim_amount, bg)` — Blends an RGB hex color toward a
  background color by a given amount. Used to construct dimmed versions of
  item colors when only some items are selected.
- `compute_majority(items_subset)` — Vectorized majority + share computation
  described in the [majority metric](#majority-metric) section.

### Section 4: App layout

The app is structured as a flat sequence of Streamlit widget calls and
data transformations, executed top to bottom on every interaction. The
order matters because filters depend on earlier widget values:

1. Page config and CSS injection.
2. Title block.
3. Load data (cached after first call).
4. **Control row 1:** Regions pills and Items pills, side by side.
5. **Control row 2:** Date range, bin rank range, position range sliders.
6. **Control row 3:** Sort mode radio and cell size controls.
7. Apply filters: build `visible_bin_indices` from the region and rank
   filters; extract `date_indices` and `pos_indices`.
8. Call `compute_majority()` on the filtered subset.
9. Apply position filter to the majority and share arrays.
10. Compute sort order based on the selected sort mode.
11. Build the Plotly colorscale, accounting for item selection.
12. Compute effective cell sizes from the slider, auto-fit checkbox, and
    visible counts.
13. Assemble customdata and text arrays for hover.
14. Build a 2×1 Plotly subplot: heatmap on top (85% height), stacked
    marginal bars on the bottom (15% height).
15. Configure axes (bin names on the y-axis reversed so the first-sorted
    bin is at top; position numbers on the x-axis; percentage ticks on
    the marginal y-axis).
16. Render the view summary block (bins, positions, date range, mode).
17. Render with `st.plotly_chart(fig, use_container_width=False)`.

### Section 5: Styling

Custom CSS injected via `st.markdown(..., unsafe_allow_html=True)` adjusts
fonts (Fraunces for the title, IBM Plex Mono for labels and tickmarks, IBM
Plex Sans for body), removes Streamlit's default border-radius on buttons,
and restyles widget labels to look like the rest of the typographic system.

### Item selection state

The Items pills group needs to be programmatically resettable by the
Select all / Clear buttons. This is implemented by:

1. Initializing `st.session_state['items_pills']` to all items on first
   load.
2. Rendering the pills widget with `key='items_pills'`, which makes
   Streamlit read from and write to that session-state key automatically.
3. The Select all / Clear buttons mutate
   `st.session_state['items_pills']` directly and call `st.rerun()` so
   the change takes effect immediately.

### Filter pipeline

The bin filter is a logical AND of region membership and rank range:

```python
in_region = np.isin(data['bin_regions'], regions_active)
in_rank   = (data['bin_ranks'] >= rank_range[0]) & \
            (data['bin_ranks'] <= rank_range[1])
visible_bin_indices = np.where(in_region & in_rank)[0]
```

`visible_bin_indices` holds the original bin indices (0–99) of bins that
passed the filter. This is then used to slice the items array:

```python
items_sub = data['items'][visible_bin_indices][:, date_indices, :]
```

producing a `(n_visible_bins, n_selected_weeks, 50)` subset on which
`compute_majority()` is called.

### Sort implementation

Each sort mode computes an `order` array — the permutation that, when
applied to `visible_bin_indices`, produces the desired row order:

```python
if sort_mode == "Similarity":
    top4 = majority[:, :4]
    keys = [''.join(f'{int(x):x}' for x in row) for row in top4]
    order = np.argsort(keys, kind='stable')
```

The hex-encoded string trick handles up to 16 item types (we have 10). For
more items, switch to a base-26 encoding using letters, or use a tuple key
and Python's `sorted()`. Stable sort ensures consistent secondary ordering.

### Color and dimming

The colorscale is rebuilt on every rerun:

```python
def effective_color(i):
    if all_or_none or i in sel_idx_set:
        return COLORS[i]
    return dim_color(COLORS[i], 0.88)

colorscale = []
for i in range(10):
    c = effective_color(i)
    colorscale.append([i / 10, c])      # step start
    colorscale.append([(i + 1) / 10, c]) # step end
```

The duplicated stops create a discrete (stepwise) colorscale where each
integer 0–9 maps unambiguously to one color. When an item is not
selected, its color is pre-blended toward the background at 88% — this
produces the dimming effect without needing per-cell opacity, which would
require switching to a per-cell trace or a more complex approach.

### Sizing logic

Streamlit doesn't expose the container's pixel width to Python. The app
uses a fixed assumption of 900 px for the position axis width (configurable
via `container_w`) and 720 px for the bin axis height (`container_h`). Adjust
these constants or disable auto-fit and use the slider directly if your
display differs significantly.

The auto-fit formula:

```python
if auto_fit:
    cell_w = max(cell_size, container_w / max(n_show_pos, 1))  # positions → columns
    cell_h = max(cell_size, container_h / max(n_show_bins, 1)) # bins → rows
else:
    cell_w = cell_size
    cell_h = cell_size

cell_w = min(cell_w, 40)
cell_h = min(cell_h, 30)
```

Plotly figure dimensions are then:

```python
heatmap_width  = int(n_show_pos  * cell_w)
heatmap_height = int(n_show_bins * cell_h)
total_width  = heatmap_width + 170   # left margin for bin name labels
total_height = int(heatmap_height / 0.83) + 60
```

### Plotly figure assembly

The 2×1 subplot is built with `make_subplots` and `shared_xaxes=True`:

```python
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.85, 0.15],
    vertical_spacing=0.025,
    shared_xaxes=True,
)
fig.add_trace(go.Heatmap(...), row=1, col=1)
for item_idx in range(10):
    fig.add_trace(go.Bar(..., orientation='v'), row=2, col=1)
fig.update_layout(barmode='stack', ...)
```

Stacked bars come from `barmode='stack'` plus 10 separate `Bar` traces
sharing the x-axis (positions) with the heatmap. The heatmap y-axis is
reversed (`autorange='reversed'`) so the first-sorted bin appears at the
top, matching natural reading order for a ranked list.

---

## Extending for real data

To use this app with real data, replace `generate_data()` with a function
that loads your data and returns a dictionary matching the schema above.

A minimal example loading from a CSV with columns
`bin_id, date, position, item, bin_rank, region` (plus optional `bin_name`):

```python
import pandas as pd

@st.cache_data
def generate_data():
    df = pd.read_csv('your_data.csv', parse_dates=['date'])

    # Establish stable orderings
    bin_ids = sorted(df['bin_id'].unique())
    dates = sorted(df['date'].dt.date.unique())
    item_to_idx = {code: i for i, code in enumerate(ITEMS)}

    n_bins, n_weeks = len(bin_ids), len(dates)
    items = np.zeros((n_bins, n_weeks, NUM_POSITIONS), dtype=np.int8)

    bin_idx = {b: i for i, b in enumerate(bin_ids)}
    date_idx = {d: i for i, d in enumerate(dates)}

    for row in df.itertuples():
        items[bin_idx[row.bin_id],
              date_idx[row.date.date()],
              row.position - 1] = item_to_idx[row.item]

    # Per-bin attributes (assumed constant across dates and positions)
    bin_attrs = df.groupby('bin_id').agg(
        bin_rank=('bin_rank', 'first'),
        region=('region', 'first')
    ).loc[bin_ids]

    return {
        'items':       items,
        'bin_ranks':   bin_attrs['bin_rank'].to_numpy(),
        'bin_regions': bin_attrs['region'].to_numpy(),
        'bin_names':   np.array([str(b) for b in bin_ids]),
        'dates':       list(dates),
    }
```

The CSV upload in the sidebar handles this automatically. Supported columns:

| Column | Required | Notes |
|---|---|---|
| `bin_id` | ✓ | Identifier; used as display name if `bin_name` absent |
| `date` | ✓ | Any format parseable by `pd.to_datetime` |
| `position` | ✓ | Integer rank within the bin |
| `item` | ✓ | Any string code — up to 10 unique values |
| `bin_rank` | ✓ | Global rank of the bin (integer) |
| `region` | ✓ | One of: NA, SA, EU, AS, CN, AU |
| `bin_name` | — | Optional human-readable display name for the bin |

Item codes can be anything (product names, category labels, emoji, etc.);
you are no longer restricted to the synthetic 10-item vocabulary.

---

## Dependencies

| Package    | Minimum version | Purpose                                          |
| ---------- | --------------- | ------------------------------------------------ |
| streamlit  | 1.40            | Web app framework; `st.pills` requires ≥1.40.    |
| plotly     | 5.18            | Heatmap and subplot rendering.                   |
| numpy      | 1.24            | Vectorized array operations and majority math.   |

Optional fonts (loaded from Google Fonts at runtime, no install required):
Fraunces (display), IBM Plex Mono (labels), IBM Plex Sans (body).
