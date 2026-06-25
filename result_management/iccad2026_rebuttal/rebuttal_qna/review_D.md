[R-D] Response to Reviewer D

Q2/W2 — Suggested baselines.

Prior ML-based cell-delay prediction work has mainly compared model architectures rather than formulating characterization as a few-shot adaptation problem over unseen cell topologies. Our framing adapts MAML to this EDA setting by defining each cell/arc as a task and evaluating K-shot adaptation for cross-topology delay-LUT regression. We therefore selected Table 4's comparison points to show how the proposed training/adaptation formulation improves over conventional architecture-focused baselines, while isolating the three engineering levers we contribute: the current-path graph, meta-learned initialization, and selective-Adam adaptation.

Several of R-D's suggested categories are already reflected in the paper's comparison points. Aadam [11] covers the "Adam-only adaptation without meta-learning" category — using hidden = 256 vs. our MLP_MAML's hidden = 40, so it is a stronger MLP-side comparison than a matched-size random-init baseline. GCN baseline [17] covers the "pretrained GCN fine-tuning without MAML" category, sharing GCN_MAML's architecture so the comparison isolates the MAML lever alone.

The remaining categories — from-scratch training on few samples, and classical feature-based regression — we agree are useful scoping checks. We did not place them in the primary table because the main table was designed to isolate our three levers. A preliminary check (global ridge with polynomial-degree-2 interactions on the same Commercial train/test partition, with and without TAMEL's Stage 1 affine alignment grafted on) showed they sit more than an order of magnitude below Aadam in cell-delay accuracy, so they would not affect the ranking. We will add this rationale to §4.2 of the camera-ready.
