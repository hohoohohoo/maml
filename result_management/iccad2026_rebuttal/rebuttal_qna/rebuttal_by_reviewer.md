# Rebuttal — Paper #642 TAMEL

We thank the four reviewers for their careful reading and constructive
feedback. The remarks on sequential-cell modeling, pessimism control,
intuitive metrics, dataset breadth, and graph-extraction transparency
prompted new experiments whose results we summarize per reviewer below.
All new evidence is cross-referenced to the supplementary material; no
claim is made beyond what the experiments support.

---

## Reviewer A

### Q1 / W1 — Sequential-cell modeling and metastability-bisection concern

> **Q1.** "How exactly does your static current-path graph represent
> the internal state-retention feedback loops of a D-Flip-Flop? …
> how does your model capture the non-linear metastability bisection
> search required for these constraints without exploding the graph
> dimension? Or did you employ a 'mixed library' approach where DFF
> metrics were simply retained from the golden SPICE library?"

**(i) Why the current-path graph captures the keeper feedback loop.**
For C2MOS-style D-FFs (e.g., DFCNQD1), the CP→Q and CDN→Q arcs propagate
through clocked tristate inverter stages while the cross-coupled keeper
inverter resists the incoming change. The keeper-side devices participate
in two physically distinct ways: (a) a **contention current** at the
shared storage net, and (b) a **gate-capacitive load** on that net.
Our graph captures both — the keeper is structurally connected via
D/S edges through the shared storage net, its gate-capacitive load
appears in the storage net's width-sum feature, and the **loop-closing
pass** walks the feedback edge and adds the keeper's **opposite-polarity
transistors** into the same pull-up / pull-down subgraphs as the
forward driver. Those are precisely the devices that turn on during
keeper contention, so the GNN sees the fighting current path rather
than only its loading effect. The construction is filtered by
`visited_target_nodes` to prevent infinite recursion through the
feedback edge.

**(ii) Why the model does not need to perform metastability bisection
search at inference.** The bisection-search non-linearity that
characterizes setup/hold occurs **at lib-characterization time**,
inside SPICE: the tool iterates data-to-clock skews until the CK→Q
curve crosses the (1 + δ)-degradation boundary, typically requiring
10–20 SPICE transient runs per LUT entry. The lib stores **one boundary
value per (slew, slew) cell**, and our model predicts that single value
by regression. The metastability blowup near the boundary is therefore
amortized into the training label set — our forward pass never enters
the metastable region, so no graph expansion is required to represent
the search itself. The voltage/slew dependency of the boundary value
is non-linear but of the same character as the cell-delay LUT's
voltage/slew dependency (which the GNN already fits well).

**(iii) No graph-dimension explosion.** The constraint LUTs share
exactly the same graph topology as the delay LUTs — same nodes, same
edges, same 11-D node features, same 64×2 + 256×2 conv/FC architecture.
The only adjustment is normalization-side: the constraint LUT's second
axis is a constrained-pin slew rather than an output load, so we alias
`output_load → input_slew` in `norm_stats` for constraint mode (a
1-line bypass, no model retraining). The same per-task selective-Adam
adaptation with 5-shot support and the same v6 sign-flip recipe apply.

**(iv) No mixed-library borrowing.** The DFF metrics in our synthesis
benchmark were **predicted by TAMEL**, not retained from the golden
lib. On three sequential cells held out from training (DFCNQD1,
SDFSNQD0, SDFCSNQD1) at corner FF / 0 °C, after one round of
post-rebuttal lib re-characterization (0.1 ps SPICE step to remove a
low-V convergence artifact identified during the rebuttal experiments),
TAMEL produces:

| Metric                | Cell delay      | Setup            | Hold             | Recovery / Removal | Non-seq setup / hold |
|---|---:|---:|---:|---:|---:|
| **NRMSE**             | 0.28 – 0.30 %   | 2.48 – 3.31 %    | 2.70 – 3.90 %    | 0.44 – 2.82 %       | 3.48 – 3.49 %         |
| **Per-task RMSE**     | 0.51 ps         | 1.15 – 1.39 ps   | 1.20 ps          | ≤ 1.18 ps           | 0.78 ps               |
| **Tasks evaluated**   | 7 350 – 80 850  | 450 – 2 700      | 450 – 2 700      | 225 – 1 800         | 7 200                 |

The constraint-LUT NRMSE is ~10× higher than cell-delay NRMSE, but the
**absolute RMSE remains within 1.0 – 1.4 ps** — well within the typical
clock-uncertainty budget (50 – 100 ps) of a 28 nm sign-off. The NRMSE
gap reflects the smaller absolute magnitude of constraint values
(≈ 10 – 30 ps versus 200 – 800 ps for delays), not a model-side
non-linearity gap. The paper's original scope statement —
"setup/hold excluded, deferred to future work" — is therefore tightened
in the camera-ready: setup/hold are *predicted and reported* with the
caveat that their NRMSE is higher than the delay-LUT primary claim.

### Q2 / W2 — Pessimism control / safety bound

> **Q2.** "To prevent silicon timing failures, can the TAMEL loss
> function be adjusted to enforce a strictly pessimistic boundary
> (predicting slightly higher delay) during few-shot adaptation?"

We agree the unconditional NRMSE statistic does not characterize the
sign-of-error distribution. In §S-Pessimism we report two complementary
recipes that bias the model to over-predict (safe direction for delay
sign-off):

1. **`safe_eps`** — a fixed positive shift added to the support-set
   target during inner-loop adaptation, in normalized units. Predictions
   end up shifted up by ≈ safe_eps × y_std × grad in raw units; the
   loss landscape is unchanged so converged accuracy is preserved.
2. **Pinball / quantile loss `pinball_tau > 0.5`** — replaces the
   inner-loop MSE with a τ-quantile loss; the minimizer is the
   τ-quantile of y, so ≈ τ fraction of predictions land above the
   target.

Empirically (Table S-Pessimism), `safe_eps ∈ [0.02, 0.10]` and
`pinball_tau ∈ [0.7, 0.9]` reduce the under-prediction fraction from
~50 % to **11 – 25 %** across the eight TSMC (cell × mode) buckets,
with the median accuracy cost staying within 9 – 11 percentage points
NRMSE. Both are inner-loop-only and can be toggled per-cell at
adaptation time — no retraining of the meta-init is required.

### Other weaknesses

**W3 — Meta-training overhead.**
The meta-training phase is amortized across the lifetime of the lib
characterization budget. Table 6 (Sec. 5) reports the SPICE-equivalent
break-even after 13 unseen cell-topology characterizations on TSMC 28 nm
(a typical library refresh adds 50 – 200 new topology variants per
PDK refresh), so the offline meta-training cost is recovered in less
than one library cycle. The few-shot inference path itself remains
under 100 ms per cell on a single GPU, two orders of magnitude faster
than the SPICE bisection that drives the same setup/hold
characterization.

---

## Reviewer B

### Q1 / W1 — Model-comparison fairness and MLP ≈ GCN under MAML

> **W1.** "The input parameters of the models being compared are not
> consistent."
> **Q1.** "The accuracy of the MLP and GCN methods with MAML added is
> remarkably similar. The authors need to explain the reasons for this
> phenomenon, clarify whether MAML is responsible for this variation,
> and elucidate under what conditions MAML optimizes which models most
> effectively."

We address the two together because they share a common root: **all
four models in §3.3 receive identical input parameters** — the same
decoupled transistor-level process features (T_ox, DIBL, SCE in place of
an aggregated V_th), the same per-task PVT triple, and the same per-arc
current-path edge structure when topology is consumed. The
architectural distinction is solely the function approximator: a
2-layer MLP that consumes flattened per-arc features, vs. a 2-layer
GCN that consumes the same features over the current-path graph.

The MLP ≈ GCN convergence under MAML follows from this design choice.
Once the meta-learner equalizes the inductive biases of the two
backbones during the few-shot adaptation phase, the residual accuracy
gap collapses because the **physical signal is already in the features**
— the GCN's structural prior buys progressively less once both networks
see the same transistor-level decomposition. The same convergence
appears with permutation-invariant set-encoder baselines on this
dataset; we treat this as a positive signal that the **meta-learned
initialization is the dominant lever**, not the backbone choice. A
backbone-agnostic claim is also why we report both MLP and GCN variants
in every table.

In our ablation (Fig. 4), removing MAML while keeping the GCN backbone
costs 3.7 – 6.4× NRMSE on TSMC and 3.6 – 5.2× on ASAP7, while removing
the current-path graph while keeping MAML costs only 1.4 – 2.1×. MAML
therefore matters more than backbone, which is consistent with the
MLP+MAML ≈ GCN+MAML observation. MAML helps most when (a) tasks
share a transferable physical prior — true here, since every cell is
characterized over the same PVT axes — and (b) the support set is
small (5 shots), so the meta-init's quality dominates the loss
landscape over the inner-loop fine-tuning.

### Q2 / W2 — Intuitive metrics (MAE, MAPE, WNS, TNS)

> **W2.** "The comparison of NRMSE is not intuitive and requires
> meaningful metrics for effective comparison."
> **Q2.** "It is advisable to include intuitive metrics such as WNS,
> TNS and MAE to demonstrate the method's practical effectiveness."

We agree and have re-derived the same evaluation buckets used in the
paper with two complementary physically interpretable metrics:

| PDK | Mode | AADAM | MLP+MAML | GCN baseline | **GCN+MAML (ours)** |
|---|---|---|---|---|---|
| Cell-delay **MAE (ps)**, geomean across held-out cells | | | | | |
| ASAP7 | Interp. | 1.700 | 0.497 | 4.432 | **0.462** |
| ASAP7 | Extrap. | 1.594 | 0.907 | 5.259 | **0.711** |
| TSMC  | Interp. | 3.587 | 0.651 | 11.130| **0.538** |
| TSMC  | Extrap. | 2.152 | 0.979 | 9.838 | **0.950** |
| Cell-delay **MAPE (%)**, same buckets                  | | | | | |
| ASAP7 | Interp. | 1.776 | 0.562 | 5.467 | **0.628** |
| ASAP7 | Extrap. | 1.923 | 1.025 | 5.727 | **0.995** |
| TSMC  | Interp. | 2.718 | 0.620 | 8.277 | **0.448** |
| TSMC  | Extrap. | 1.860 | 0.874 | 7.475 | **0.724** |

Under 1 % MAPE on every (PDK × mode) bucket, with **0.46 – 0.95 ps**
absolute cell-delay error on cells held out from training.

For path-level WNS / TNS / ΔPath Delay we updated Table 5 with the
post-rebuttal STA run at FF / 0 °C across the six benchmarks:

| Cells       | Critical Path SPICE (ns) | Critical Path TAMEL (ns) | ΔPath Delay (ps) | ΔWNS (ps) |
|---|---:|---:|---:|---:|
| aes_ip      | 0.4782 | 0.4757 |  8.81 | 18.27 |
| s5378       | 0.3649 | 0.3647 |  0.55 |   —   |
| s38584      | 0.4068 | 0.4084 |  7.35 |  8.29 |
| picorv32    | 0.4448 | 0.4463 |  2.10 | 13.13 |
| vga_enh_top | 0.4082 | 0.4088 |  3.26 |   —   |
| darkriscv   | 0.4732 | 0.4722 | 20.00 | 22.05 |
| **AVERAGE** | **0.4293** | **0.4293** | **7.01** | **15.44** |

The average ΔPath Delay drops to **7.01 ps on a 429.3 ps path (1.6 %)**
and average ΔWNS stays within **15.44 ps**, both well inside the
clock-uncertainty margins commonly used for 28 nm sign-off. The
single-corner ΔTNS averages 14.08 ns across all 44 violation points
(supplementary §S-STA-Q2, Table S-STA).

### Other weaknesses

**W3 — Novelty / GCN+MLP+MAML "commonly used models".**
The methodological novelty rests on three jointly designed pillars,
none of which alone is sufficient on this task:
(i) **decoupled transistor-level features (T_ox, DIBL, SCE)** —
replacing a single aggregated V_th with the three microscopic drivers
makes voltage-dependent effects mathematically separable from aging;
removing this decoupling collapses the cross-topology NRMSE from
0.43 % to 2.7 % on TSMC interpolation (ablation row 2 in Fig. 4);
(ii) **current-path-aware topology graph** — edges encode the union of
conductive routes for each (output × direction) timing arc, and the
loop-closing pass adds opposite-polarity keeper transistors so
sequential-cell contention is captured (the graph is not the same as
the full-netlist GCN baseline, which includes every D/S edge
irrespective of conductivity under the timing arc);
(iii) **cross-topology meta-learning framing** — MAML is recast for
per-task regression over cell topologies; the selective-Adam
adaptation rule is dictated by this framing and does not transfer
directly from the "MAML + GCN" combination used for classification
in prior work. The camera-ready makes the three pillars explicit and
ties each to its ablation row.

**W4 — Dataset too small.**
The training and held-out splits across both PDKs cover **~847 million
delay/transition samples** on training and **~149 million held-out
samples**, with hold-out enforced separately by cell topology and by
drive-strength variant. Table S-Scale enumerates per-cell sample
counts; the dataset is large relative to prior single-node /
no-hold-out benchmarks in this line of work.

**W5 — Table 5 shows only TT corner.**
The updated Table 5 in the camera-ready adds the FF / 0 °C corner
sweep tabulated under Q2 above. We acknowledge in the text that the
extreme low-V corner of FF / 0 °C (V_dd ≤ 0.74 V) crosses the
synthesizer's path-selection sensitivity threshold — the worst-case
ΔCPD at darkriscv / 0.74 V (59.7 ps) is dominated by the synthesizer
flipping which path is the critical one, not by per-LUT mis-prediction.
Cross-STA on identical netlists (in progress) is expected to isolate
the pure lib-side effect to ≤ 2 ps; we will report the cross-STA
numbers in the supplement.

**W6 — Training/testing time long for "small" dataset.**
The 500× SPICE speedup (Fig. 6) compares characterization wall-clock,
not training wall-clock, and the meta-training overhead is amortized
over the library characterization budget as noted under Reviewer A-W3.
With ~847 M training samples, the dataset is not small; the
characterization-per-cell time including 5-shot adaptation remains
under 100 ms per cell on a single GPU.

---

## Reviewer C

### Other weaknesses

**W — Current-path graph for sequential cells / metastability / risk of
dimensional explosion.**
Reviewer C's weakness restates Reviewer A's W1/Q1 in stronger terms.
The graph-mechanics argument (loop-closing pass, opposite-polarity
keeper transistors, no inference-time bisection) and the empirical
evidence (3 sequential cells, NRMSE 2.5 – 3.9 %, RMSE 1.0 – 1.4 ps for
setup/hold) under Reviewer A apply directly. We summarize the
conclusion for Reviewer C: the graph is **structurally identical** for
combinational delays, sequential delays, and sequential constraint
LUTs — no dimensional explosion is required, and the empirical
setup/hold accuracy is within typical clock-uncertainty margins for
28 nm sign-off.

---

## Reviewer D

### Q1 / W1 — Current-path graph extraction algorithm

> **W1.** "The paper uses timing-arc-specific current-path graphs, but
> the exact extraction algorithm from the transistor netlist is unclear."
> **Q1.** "Could the authors clarify the exact algorithm used to extract
> these current paths from the transistor-level netlist? In particular,
> how are stacked/parallel transistors, reconvergent paths, internal
> nodes, and input-state-dependent conducting paths handled?"

We provide a step-by-step extraction algorithm and a worked example in
supplementary §S-GraphExtract; key invariants:

1. **Stage-wise backward traversal.** For each (output pin, transition
   direction) pair, start at the output net and perform a
   polarity-restricted DFS — **PMOS-only from VDD for rise, NMOS-only
   from VSS for fall** — that finds every conductive source-drain route
   to the rail under the timing-arc's `when` condition. The
   non-external gate nets of those transistors become the **target
   nets of the previous (input-side) stage**, with MOS polarity and
   supply rail alternating accordingly. **Stacked** transistors appear
   as multi-hop S/D edges along the path; **parallel** transistors
   appear as alternative DFS branches. **Reconvergent** paths are
   aggregated by **union of edges** — every conductive route is
   preserved — so the GNN sees the full conductive set rather than a
   single representative path.
2. **Internal nodes.** Every internal net becomes a graph node whose
   feature is the **sum of widths** of all transistors whose gate ties
   to that net (total gate-capacitance load). VDD/VSS are typed nodes;
   transistor nodes carry polarity ±1 in column 2 of the feature
   vector.
3. **Input-state dependence.** Inputs are not separate nodes; their
   state is fixed by the LUT's `when` clause, which deterministically
   selects which transistors are conducting and therefore which DFS
   branches survive. The extracted graph is per-arc and per-direction
   for this reason.
4. **Sequential cell feedback.** An opt-in **loop-closing pass** walks
   the feedback edge of cross-coupled storage nets and adds the
   keeper's opposite-polarity transistors into the same pull-up /
   pull-down subgraphs as the forward driver (see Reviewer A-W1/Q1),
   gated by a `visited_target_nodes` filter to prevent infinite
   recursion.

The extraction is fully automatic: we attach pseudocode (Algorithm 1 in
the supplement) and an illustrative figure for a 2-input NAND, an
AOI21, and the DFCNQD1 D-FF.

### Q2 / W2 — Comparison against additional baselines

> **W2.** "Compare only with one previous work."
> **Q2.** "Could the authors explain why other baselines and previous
> works were not included?"

The original submission compared against AADAM as the single direct
prior work because it shares the same per-cell delay-LUT regression
framing. We add the following baselines in the camera-ready
(supplementary §S-Baselines):

1. **GCN trained from scratch with 5 support points** — fits only the
   support points, NRMSE > 8 % off-support.
2. **Pretrained GCN fine-tuned without MAML** (single inner step).
3. **Feature-only ridge regression** on the same decoupled
   transistor-level features.
4. **Adam-only adaptation without meta-init** (random initialization,
   40 inner steps).

Across both PDKs, the meta-learned initialization remains the dominant
lever — non-meta baselines achieve 4 – 12× worse NRMSE on the
held-out-topology buckets.

---

## Summary

The post-rebuttal experiments tighten three claims:
(i) sequential-cell setup/hold are within the same current-path graph
and the same model — no graph expansion, no per-inference bisection;
(ii) safe-direction inference is available without retraining, at
9 – 11 pp NRMSE cost;
(iii) the FF / 0 °C synthesis-level error drops to **7.01 ps mean
ΔPath Delay / 1.6 %**, with the worst case (darkriscv) explained by
synthesizer path-selection sensitivity rather than per-LUT
mis-prediction. We thank the reviewers again for the questions that
prompted these additional results.
