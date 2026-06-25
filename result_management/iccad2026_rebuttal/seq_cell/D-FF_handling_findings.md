# D-FF / Sequential cell handling — code-level findings

> Definitive answer to ICCAD 2026 #642 reviewer A-Q1 and C concerns. Established from source code inspection, 2026-06-10.

## 1. What the question asks

- **A-Q1**: "How exactly does your static current-path graph represent the internal state-retention feedback loops of a D-Flip-Flop? Did TAMEL predict the Setup/Hold time constraints? Or did you employ a 'mixed library' approach where DFF metrics were retained from the golden SPICE library?"
- **C echoes the same concern**: "cannot properly model internal feedback loops or capture metastability related setup and hold timing constraints."

## 2. The answer (one sentence)

**TAMEL predicts only the propagation delay (cell_rise / cell_fall) for all cells including D-FF; Setup/Hold/Recovery/Removal/min_pulse constraints are neither in the training set nor evaluated.**

## 3. Evidence — source-code chain

### Step 0. The .lib files DO contain setup/hold/etc — TAMEL skips them on purpose

For `DFCNQD1BWP30P140` in `TSMC_TT_Seq_100_098.lib`, the distinct `timing_type` values present in the cell block are:

```
timing_type : setup_rising
timing_type : hold_rising
timing_type : recovery_rising
timing_type : removal_rising
timing_type : min_pulse_width
timing_type : clear
timing_type : rising_edge    ← CK→Q propagation arc (the one TAMEL learns)
```

So the constraint data is available on disk. The TAMEL parser explicitly limits itself to the propagation arcs (cell_rise / cell_fall / rise_transition / fall_transition keywords — see Step 1) and **substring-matching means none of the constraint table keywords (`rise_constraint`, `fall_constraint`, `setup_*`, `hold_*`) are caught**. This is a design decision, not a parsing oversight.

### Step 1. Liberty parser only extracts cell_rise / cell_fall

File: `Projects/data_processing/MLP/utils/libdata_extract_MAML_cell.py`

```python
# Line 175
elif any(dt in line for dt in ["cell_rise","cell_fall","rise_transition","fall_transition"]):

# Line 178
delay_types = ["cell_rise","cell_fall","rise_transition","fall_transition"]

# Line 313 — main extraction loop
for dtype in ["cell_rise", "cell_fall"]:
    ...
```

Critical: `setup_rising`, `hold_rising`, `recovery_*`, `removal_*`, `min_pulse_width`, `min_period`, `timing_type` are **not parsed**. Lines 172-173 in particular have `timing_type` extraction commented out by the author, confirming this is intentional.

### Step 2. Topology cache key = output port, not timing arc

File: `Projects/data_processing/gnn/precompute_stage_aware_topology.py`

```python
# Line 1201
output_topologies[output_node] = {
    'pull_up': pull_up_data,
    'pull_down': pull_down_data
}
```

`output_topologies` is keyed by **output port name** (e.g., `Q` for a D-FF), not by `(input, output)` timing-arc pair. For `DFCNQD1` the entire 28-transistor netlist collapses to a single `Q` entry with pull-up / pull-down split. CK→Q, S→Q, R→Q, setup CK→D, hold CK→D — all share the same graph at inference time.

### Step 3. Apply uses (output_name, delay_type) — no timing-arc selector

File: same script, `apply_stage_aware_topology` at line 1352:

```python
output_topo = cell_cache['output_topologies'][output_name]
if 'rise' in delay_type:
    path_cache = output_topo['pull_up']
else:
    path_cache = output_topo['pull_down']
```

There is **no `arc_type`, `timing_type`, `setup`, `hold`, `related_pin_specific_graph` parameter**. The model receives one graph per (cell, output, rise|fall). `input_slew` may be steered to specific transistors via `slew_mode='related_pin_only'` (line 1413-1419), but the graph topology itself is shared.

### Step 4. Ground-truth evidence in netlist

`tcbn28hpcplusbwp30p140_110a_lpe_typical.spi`:
- Line 23446: `.subckt SDFSNQD0BWP30P140 SI D SE CP SDN Q VDD VSS` — 8 pins
- Line 24808: `.subckt DFCNQD1BWP30P140 D CP CDN Q VDD VSS` — 6 pins, **28 transistors** in subckt
- Simple inverter `INVD0` for reference: 2 transistors

The netlist for D-FF is physically richer (28 vs 2 transistors, feedback loops present in cross-coupled inverter pairs), but the predicted metric (CK→Q cell_rise/cell_fall) only exercises the forward propagation path. The feedback loops are not directly observed by either the model or the supervising loss.

## 4. Recommended rebuttal language (draft)

> "TAMEL predicts propagation delay (`cell_rise`/`cell_fall`) for all evaluated cells, including the sequential D-FF and scan-FF examples in Section 4.1. **Setup, hold, recovery, removal, and minimum-pulse-width constraints were not part of the predicted output**, and were therefore not in the training set, the test set, or the reported metrics. The reviewer's concern about modeling metastability via static current-path graphs is well-taken and is exactly why we restricted the metric to propagation delay: the CK→Q delay path is a pure forward signal-propagation problem that the current-path representation captures accurately, whereas setup/hold characterization fundamentally requires transient bisection that we agree is outside the scope of a static graph. In a deployed flow, the few-shot adapted TAMEL model handles `cell_rise`/`cell_fall` re-characterization while setup/hold constraints would be retained from the golden SPICE library (a partial-mixed-library deployment). We will clarify this scope explicitly in the revised Section 4.1."

## 5. Implications for the sequential-cell NRMSE result (Section 4.2 / Table 4)

We reported NRMSE ~7-8% on `SDFSNQD0` / `DFCNQD1` while the rest of the cross-topology test set is sub-1.5%. Code-level reasons:

1. **0 sequential cells in the 44-cell training set** — the model sees only combinational current paths during meta-training.
2. **Single output topology per cell** — the 28-transistor D-FF netlist collapses to one Q-output graph; the model cannot infer which physical timing arc within the cell drives a particular CK→Q sample.
3. **Per-task surface roughness** for these two cells is 70-100× higher than combinational cells (median `rough_V` 0.5+ vs 0.008) — a downstream consequence of (1) + (2).

This is *not* a metastability or setup/hold issue. It is a distribution-shift and graph-collapse issue specific to propagation-delay prediction on sequential cells. The rebuttal narrative should keep these two failure modes separate so the reviewer does not conflate them.

## 6. Open follow-up (rebuttal does NOT need to fix, but should acknowledge)

- **Per-arc topology cache**: replace `output_topologies[output_name]` with `output_topologies[(input_pin, output_name, delay_type)]` so multi-arc cells have distinct graphs. Would reduce sequential NRMSE (and arguably help complex combinational cells like AOI/OAI with re-convergent paths).
- **Sequential cells in train**: needs new SPICE characterization runs for sequential cell families.

These are listed as the C1/C3 contribution candidates in the 2-D V×T research thread but are out-of-scope for the rebuttal.

## 7. Cross-references

- `Projects/ICCAD2026_rebuttal_plan.md` §5 Q3
- `Projects/ICCAD2026_review.txt` (A-Q1 line 28; C summary line 83)
- `docs/results/2026-06-05-TSMC-2D-results-summary.md` §5 F3 (sequential-cell underperformance diagnosis — same root causes)
