### What you're looking at

A heatmap where each **row** is a %%bin_term%%, each **column** is a ranked position,
and each **cell** is coloured by the %%item_term%% occupying that slot.
When you select a date range spanning multiple snapshots, each cell is resolved
according to the active **Method** (see the Method section in Row 1).
Hover any cell for exact values.

Below the heatmap is a stacked bar showing the %%item_term%% distribution
across visible %%bin_term%%s at each position (interpretation varies by method).

[How the heatmap, bar, and drill-down connect →](%%VIZ_GUIDE_URL%%)

---

### %%Filter_term%% — Row 1 (optional, leftmost)

Shown only when your data has a `filter` column. One value is active at a time;
selecting a pill restricts the heatmap to rows with that provenance label — only
those rows drive cell colours, shares, and the bar chart. Bins that carry no data
for the selected value are hidden. Rename this label via **Labels → Filter attribute**.

---

### %%Segment_term%%s — Row 1

Toggleable pills. All values are selected by default; deselecting one hides every
%%bin_term%% whose %%segment_term%% is not selected. Use **all** / **none** buttons to
select or clear in bulk. When a filter is active, only segments present in that
filter's data are shown.

---

### %%Item_term%%s — Row 1

Toggleable pills. All %%item_term%%s are selected by default; controls *highlighting*, not
filtering. Deselected %%item_term%%s are dimmed; selecting **none** dims all %%item_term%%s.
Gray %%item_term%%s (beyond the top-10 distinct colours) are always visible at full colour
in the heatmap but are not shown as pills. Use **all** / **none** buttons to
select or clear in bulk.

---

### Method — Row 1 (rightmost)

| Method | What it does |
|---|---|
| **Majority** | For each date, the plurality winner (most observations); across dates, the %%item_term%% that won most dates is shown. |
| **Abs. Majority** | Per date: the plurality winner wins only if it holds ≥ 50 % of that day's observations; otherwise that day's value is **VARIOUS**. Cross-date aggregation is identical to Majority. |
| **Weighted** | Pools all observations across the date range; %%item_term%% share = total N\_item ÷ total group\_N. Winner = highest share. |

[Visual guide to all methods →](%%METHOD_GUIDE_URL%%)

---

### Ranges — Row 2

**Date** · Dual-handle slider. A single date shows that snapshot exactly;
a wider range triggers the selected Method's aggregation logic.

**%%Bin_term%% rank range** · Filter rows by global rank.

**Position range** · Hide columns outside the chosen window.

---

### Sort — Row 3

| Mode | What it does |
|---|---|
| **Index** | Alphabetical by %%bin_term%% ID — stable baseline |
| **Similarity** | Groups %%bin_term%%s that share the same %%item_term%%s at positions 1–4 (default) |
| **%%Bin_term%% Rank** | Ascending by global rank |
| **Top-rank** | Groups %%bin_term%%s sharing the same %%item_term%% at position 1; ties resolved by position 2, 3, … |
| **Selected Share** | Ranks %%bin_term%%s by how many visible positions are held by selected %%item_term%%s |

[Visual guide to all sort modes →](%%SORT_GUIDE_URL%%)

---

### Drill-down

Use the drop-down selectbox below the chart to select a %%bin_term%%. A time-series
heatmap appears below, showing how its %%item_term%% mix evolved across every date
in the selected range. Hover shows the %%item_term%%'s share for that specific date.

---

### Uploading your own data

Open the **sidebar** and upload a CSV with these columns:

| Column | Required | Notes |
|---|---|---|
| `bin_id` | ✓ | Display name |
| `date` | ✓ | M/D/YYYY format (single or double-digit month/day) |
| `position` | ✓ | Ranked position within the %%bin_term%% |
| `item` | ✓ | Any string label |
| `bin_rank` | ✓ | Global rank of the %%bin_term%% |
| `segment` | ✓ | Grouping / filter attribute |
| `filter` | optional | Bin-level filter label; shown as single-select pills above the segments row. |
| `N_item` | optional | Observation count for this %%item_term%%. Defaults to 1 per row if absent. |

One row per unique [bin\_id, date, position, segment, **item**].
`group_N` (total observations per cell) is computed internally as `sum(N_item)` across %%item_term%%s sharing the same [bin\_id, date, position, segment] key.
`bin_rank` is bin-level metadata (one value per %%bin_term%%), not part of the aggregation key.
