# Rebuttal Writing Rules — TAMEL (ICCAD 2026)

Writing rules used in this rebuttal cycle. Future edits should re-check compliance
before any merge. Order is priority order: rule 1 is most important.

---

## Rule 1 — Non-aggressive tone

Reviewer questions are good-faith. Frame responses as clarifications or
acknowledgments, not as counter-arguments. Avoid making the reviewer wrong;
make our scope explicit instead.

**Avoid:**
- "We respectfully clarify that the reviewer is wrong because..."
- "Off-the-shelf X fails / does not work"
- Adverbs that read defensive: "in fact", "strictly", "trivially", "obviously"
- Verbs that imply judgement of the suggestion: "inappropriate", "unfounded"

**Prefer:**
- "We agree these are useful scoping checks. We chose ... because ..."
- "Several of R-X's suggestions are already reflected in the paper's comparison points."
- "MAML provides the core generalization lever; our skills extract that maximally."
- "In our judgment, ..." / "In our view, ..." (only when needed to mark opinion)

Even when defending a methodological choice, lead with what we did and why,
not with what the reviewer got wrong.

---

## Rule 2 — One concrete final number, not a list

When making an empirical claim, end with a single concluding number rather than
a range, a sequence, or a per-bucket list. Reviewers skim; give them the one
number they need to remember.

**Avoid:**
- "NRMSE drops from 50% to 25%, then to 10%, then to 5% with stronger knobs"
- "Alignment alone reaches 1.67% / 3.19% NRMSE on commercial / ASAP7 in 0.40 / 1.13 hr; selective Adam reaches 0.69% / 0.99% in 1.96 / 6.11 hr ..."

**Prefer:**
- "NRMSE settles at ~10% on average" — pick the best operating point
- "Paper Table 6 quantifies this split" — let the table carry the numbers

If two PDKs unavoidably split the value, present as a single pair "TSMC / ASAP7"
rather than a four-cell matrix.

For values already in a paper table, cite the table location ("paper Table X")
instead of re-listing the cells.

---

## Rule 3 — One thesis per paragraph

Each paragraph should advance one clear point that the reader can summarize in
one sentence. Mid-paragraph topic jumps force the reader to re-read.

**Pattern:**
1. Open with the thesis as a title-style first sentence (e.g., "W4 — The dataset is not small.")
2. Body provides the supporting evidence
3. Close with a single concluding pointer (cite a table / section) or omit a closing

If a paragraph is doing two things at once (e.g., explaining a mechanism AND
defending against a critique), split into two paragraphs.

Test before submitting: *"What is the one thing this paragraph says?"* If you
can't answer in one sentence, the paragraph needs to be split or trimmed.

---

## Rule 4 — Reference, don't repeat

Content already in the paper goes by location-only reference; reserve rebuttal
prose for *new* clarifications.

**Avoid:**
- "Paper Table 6 shows alignment-only NRMSE at 1.67% / 3.19% on commercial / ASAP7..."
- "Section 3.4 explains that adaptation is two-stage with alignment first, then..."

**Prefer:**
- "Paper Table 6 quantifies this split"
- "Adaptation is two-stage (Sec. 3.4): alignment first, then refinement"

Reference targets must be **stable** under camera-ready edits:
- ✓ Section number (Sec. 3.3)
- ✓ Table number (Table 4)
- ✓ Figure number (Fig. 4)
- ❌ Line numbers (l.403–417) — these shift; do not use

---

## Cycle-specific principles (this rebuttal cycle)

### 5. Positive framing of our contribution

State what our contribution *does* before stating what prior work *fails to do*.
"MAML provides the core generalization lever; we extract it maximally on this
distribution" beats "off-the-shelf MAML fails here, we have to fix it."

### 6. Acknowledge legitimate critique before redirecting

"We agree X is useful; we chose Y because Z" is more persuasive than refusing
the premise. The acknowledgment lowers the temperature of the response.

### 7. Honest about limits — no over-claim

If we reach ~10% under-rate, say ~10%. Do not write "near-zero". Reviewers
will run the numbers; over-claims cost more credibility than honest limits.

### 8. Move per-reviewer-specific content into the per-reviewer section

Common blocks ([C1], [C2]) hold cross-cutting concerns only. Content that only
one reviewer raised (e.g., the MLP+MAML ≈ GCN+MAML mechanism for R-B Q1) belongs
in that reviewer's section, not the common block — so other reviewers do not
see content irrelevant to their questions.

### 9. Strict word budget (ICCAD 2026: 2,000 words total)

The 2,000-word limit applies to the entire rebuttal — text box and PDF
combined — per chair guidance. Tables, figures, and algorithms in the PDF
supplement do not count as prose; their captions and any explanatory paragraphs
do count. Stay 30–50 words under the limit to absorb late edits.

### 10. PDF supplement contains structurally non-prose material only

Each item is its own `.tex` fragment, referenced from the main supplement via
`\input{}`. Body prose cites supplement items by stable tag — e.g.,
`PDF [Table B-W1]`, `PDF [Algorithm 1]` — so the body and the PDF can be
edited independently.

---

## Quick checklist before submitting

- [ ] No "in fact" / "strictly" / "trivially" / "obviously" survive
- [ ] Every empirical claim ends with one concrete number
- [ ] Every paragraph has one thesis sentence at the start
- [ ] All references are stable (section / table / figure numbers, no line numbers)
- [ ] Per-reviewer content lives in that reviewer's section, not in [C1] / [C2]
- [ ] PDF supplement tags (`[Algorithm 1]`, `[Table B-W1]`, ...) all resolve to existing entries
- [ ] Combined word count (body + supplement caption prose) < 2,000
