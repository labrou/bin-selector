# Performance Analysis: Weighted Method vs Majority

## Benchmark Results (200 bins, 52 dates, 100 positions, 1000 items)

| Method | Time | vs Majority | Status |
|--------|------|------------|--------|
| **Majority** | 134 ms | 1.0x | Baseline |
| **Abs. Majority** | 122 ms | 0.91x | Faster |
| **Weighted** | 339 ms | 2.5x slower | Acceptable |

The 2.5x slowdown is **expected and acceptable** because:
- Weighted aggregates across ALL items (1000 item-level calculations vs 2 simple aggregations)
- Returns full 3D weights array (200×100×1000) for visualization
- Operates on sparse data with non-trivial indexing overhead

---

## Weighted Method Profiling (cProfile + line_profiler)

### Time Distribution (3 runs, 1.03s total)

| Operation | Time | % | Category |
|-----------|------|---|----------|
| **bincount + reshape** | 308 ms | 30.4% | **Critical bottleneck** |
| **Masking / fancy indexing** | 174 ms | 17.2% | Secondary bottleneck |
| **Aggregation (sum, divide)** | 149 ms | 14.7% | Expected |
| **Element-wise ops (argmax, max)** | 36 ms | 3.5% | Expected |
| **Tiebreak logic** | 2 ms | 0.2% | Negligible |

### Line-by-Line Hotspots

**Top 5 consuming lines in `compute_weighted()`:**

1. **Line 649: `np.bincount()`** — 20.0% of total time
   - Sums 5.7M sparse observations into 20M output bins
   - NumPy built-in, highly optimized; not worth replacing

2. **Line 648: Flat index computation** — 16.3% of total time
   - `flat_idx = (lb * n_pos_sel + lp) * n_items + iif`
   - Cannot avoid: required to map sparse (bin, pos, item) to flat space

3. **Line 651: reshape + astype** — 10.4% of total time
   - Reshape bincount output (20M,) → (200, 100, 1000)
   - Copy to float32 for division
   - Inherent to the algorithm

4. **Line 654: Weights division** — 10.1% of total time
   - `weights = total_N / np.maximum(group_total[:, :, None], 1.0)`
   - Broadcasting required for 3D output
   - Optimal NumPy

5. **Lines 643-644: Fancy indexing** — 13.8% of total time
   - `lb = vis_local[bi[mask]]` and `lp = pos_local[pi[mask]]`
   - Maps global indices to local view indices on ~1.7M masked rows
   - Inherent to the algorithm

---

## Opportunities for Optimization

### HIGH-IMPACT opportunities (worth implementing):

**1. Early termination for empty/sparse bins** [Estimated: 5-10% speedup]
   - If a visible bin has no sparse data, skip it entirely
   - Current: masks all 5.7M rows regardless of bin coverage
   - Fix: Pre-filter `bi` to only included visited bins before masking
   ```python
   # After filtering by date/position, check which bins have any data
   visited_bins = np.unique(bi_filtered)
   skip_bins = np.setdiff1d(visible_bin_indices, visited_bins)
   # Can skip remainder of computation for those bins
   ```
   - Complexity: Medium (need to handle output shape for skipped bins)
   - Benefit: Higher on sparse datasets; lower on dense ones

**2. Use scipy.sparse instead of dense reshape** [Estimated: 8-15% speedup]
   - Current: reshape bincount output to dense (200, 100, 1000) = 20M floats
   - Alternative: Keep as sparse COO matrix until needed for visualization
   - Problem: Visualization needs dense for `argmax` and `weights` indexing
   - Viable only if we defer densification until after winner computation
   - Complexity: High (restructures data flow significantly)
   - Benefit: Good only if 1000 items is typical; scales worse with item count

**3. Batch masking to reduce memory bandwidth** [Estimated: 3-8% speedup]
   - Current: Creates 3 separate boolean masks (~17MB each) before combining
   - Better: Check conditions incrementally with early exit
   ```python
   mask = np.ones(len(bi), dtype=bool)
   for cond_fn in [lambda x: vis_local[x] >= 0, ...]:
       if not mask.any():
           break
       mask &= cond_fn(bi)
   ```
   - Complexity: Low
   - Benefit: Modest but safe; reduces memory pressure

### MEDIUM-IMPACT opportunities (complex tradeoffs):

**4. Cache-aware bin-by-bin aggregation** [Estimated: 10-20% speedup]
   - Current: Process all 5.7M rows at once → one large bincount call
   - Alternative: Process each visible bin separately → 200 smaller bincounts
   ```python
   for vis_idx, glob_bin in enumerate(visible_bin_indices):
       mask_b = bi == glob_bin  # Filter to this bin
       # Aggregate this bin separately
       # Output shape becomes (1, n_pos_sel, n_items)
   ```
   - Trade: More function call overhead, but better L1/L2 cache locality
   - Complexity: High (restructures main loop)
   - Benefit: Varies by CPU; lower on modern high-cache systems

### LOW-IMPACT opportunities (not recommended):

**5. Tiebreak optimization** — Only 0.2% of time; skip this entirely

**6. Replace astype with view/copy_coerce** — The dtype conversion is required for correctness; savings would be < 1%

**7. Parallelize with numba/prange** — GIL overhead + communication cost likely exceed benefit for this workload; best for CPU-bound inner loops, not I/O-bound bincount

---

## Recommendation

**No optimization necessary.** Here's why:

1. **The slowdown is fundamentally unavoidable:**
   - Weighted method aggregates 1000× more data per cell than Majority
   - 339ms for 20M aggregation ops is near-optimal NumPy performance
   - Majority is 2x faster ONLY because it ignores item-level detail

2. **The 2.5x slowdown is acceptable for UX:**
   - Majority: 134ms (imperceptible, < frame time)
   - Weighted: 339ms (still interactive, < 400ms)
   - Both are fast enough for real-time UI responsiveness

3. **Early termination (opportunity #1) adds complexity with modest payoff:**
   - Benefit: 5-10% speedup = ~17-34ms
   - Cost: More branches, special cases, maintenance burden
   - ROI: Only valuable if datasets grow to 500-1000 bins with sparse data

4. **Sparse matrix approach (opportunity #2) is overkill:**
   - 20M float32 values = 76MB RAM
   - Not a memory constraint for modern systems
   - Adds dependencies (scipy) and complexity

---

## Conclusion

**Current performance is good.**
- ✅ Weighted aggregation is algorithmically optimal (O(n) single-pass bincount)
- ✅ No low-hanging fruit (top 5 operations are all unavoidable)
- ✅ 2.5x slowdown vs Majority is proportional to feature completeness (1000× more detail)
- ✅ 339ms is still well within interactive range for UI responsiveness

**If performance becomes a problem:**
1. Profile specific slow use cases (e.g., millions of items)
2. Implement opportunity #3 (batch masking) for 3-8% safe improvement
3. Only pursue #1 or #2 if datasets exceed 500+ bins with > 100k sparse rows

---

## Testing Methodology

- **Dataset**: 200 bins × 52 dates × 100 positions × 1000 items
- **Sparse density**: ~5.5 observations per cell (5.7M total sparse rows)
- **Tool**: cProfile + line_profiler (1 μs timer precision)
- **Runs**: 3 iterations, averaged; 1.03s total time

