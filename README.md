# xenium_LEA

**A read-only audit of a multi-run Xenium study.** It answers one question that
no amount of batch correction can answer for you:

> Are the technical differences between our runs separable from the biology we
> want to measure?

It does not correct batch effects, does not merge your runs into one object, and
does not modify anything inside a run directory. It produces a verdict, a set of
findings, and a report.

---

## Why

A Xenium study accumulates technical heterogeneity that is invisible in the
count matrix:

- Runs share a **base gene panel** but carry **different add-on panels**.
- Some runs used the **cell segmentation staining kit**, some used nucleus
  expansion.
- **Several sections come from the same mouse**, so sections are not independent
  replicates.

Any of these can produce a difference between your groups that looks like
biology. The dangerous case is when one of them *lines up with* your biological
contrast — if the segmentation kit was used on every AGED run and no ADULT run,
then "segmentation effect" and "ageing effect" are the same column of the design
matrix. Harmony will still produce a clean, well-mixed embedding. The ageing
signal will have gone into the correction along with the kit effect, and nothing
in the output will say so.

This tool exists to catch that before you analyse.

---

## Install

```bash
git clone https://github.com/AlexanderJais/xenium_LEA.git
cd xenium_LEA
pip install -e .
```

Python 3.10+. Dependencies are deliberately light — numpy, pandas, scipy,
scikit-learn, matplotlib, pyarrow. No scanpy, no anndata, so the audit runs even
where your analysis environment does not.

## Use

Write a manifest (copy `manifest_template.csv`):

```csv
run_id,mouse_id,section_id,condition,run_dir,sex,age_weeks
AGED_1_s1,M_A01,s1,aged,/data/xenium/aged_mouse1_sectionA,male,70
AGED_1_s2,M_A01,s2,aged,/data/xenium/aged_mouse1_sectionB,male,70
ADULT_1_s1,M_D01,s1,adult,/data/xenium/adult_mouse1_sectionA,female,25
```

Columns beyond the five required ones are treated as covariates and analysed —
add whatever you tracked.

Then:

```bash
xenium-lea audit --manifest manifest.csv --out audit_out          # seconds
xenium-lea audit --manifest manifest.csv --out audit_out --deep   # minutes
```

Open `audit_out/report.html`. It exits non-zero when an error-severity finding
is present, so it can gate a pipeline; `--no-fail` disables that.

**`mouse_id` is the one field you must supply.** Everything else — panel,
segmentation kit, software version, run date — is read from the run directories,
because declared metadata is exactly what drifts. Xenium writes nothing that
links two sections of the same animal, and without that link sections get
treated as independent replicates, which inflates significance.

---

## What it checks

| | Question | Output |
|---|---|---|
| 1 | What runs exist, and which mouse/section/condition is each? | `inventory.csv` |
| 2 | Is the base panel complete and identical in every run? | `panel_missing_base.csv` |
| 3 | Which add-on genes exist where, and what is comparable? | `panel_overlap.csv`, `safe_gene_set.txt` |
| 4 | Which runs used the segmentation kit? | `segmentation_audit.csv` |
| 5 | Is run quality comparable? | `cell_qc.csv` |
| 6 | **Are the technical factors confounded with condition?** | **`separability_verdict.json`** |
| 7 | How large is the run effect, and what drives which axis? | `batch_*_associations.csv` (`--deep`) |

### The verdict

For every technical factor, the question is not "how big is it" but "does it
vary *within* each condition?":

| Verdict | Meaning |
|---|---|
| `CONSTANT` | One level across the study. No effect is possible. |
| `PER_RUN` | A distinct value for every run. Labels the sample rather than grouping samples, so it carries no batch information. Excluded from the verdict. |
| `UNKNOWN_DOMINATED` | Unknown for most runs — the split tracks which files were uploaded, not the experiment. Excluded from the verdict. |
| `GROUPING` | Groups the runs, but there is no condition column yet to test it against. |
| `CROSSED` | Varies within every condition. Estimable, and adjustable without touching the condition contrast. |
| `PARTIAL` | Varies within some conditions but not all. Estimable, unbalanced, reduced power. |
| `NESTED` | Varies within a condition, but no level is shared across conditions. Not correctable as a fixed effect — including it absorbs the condition term — but its variance is estimable, so test condition against it as a random effect. |
| `ALIASED` | Constant within each condition, different between them. **The factor and condition are the same contrast. Not estimable, not removable.** |

Overall: `BLOCKED` if anything is `ALIASED`, `CAUTION` if anything is `PARTIAL`
or `NESTED`, `NO_CONTRAST` if fewer than two conditions are labelled, otherwise
`OK`.

Factors that move together are reported as one **equivalence class**, not as
every pair: eleven factors changing at the same boundary is one fact about the
study — usually one batch boundary wearing several names — and it means an
effect cannot be attributed to any one of them.

**Extra manifest columns are treated as covariates** — sex, age, surgery batch,
anything you tracked. They are checked against the technical factors as well as
against condition, because a covariate that is *indistinguishable from a
technical factor* removes a biological question even when the primary contrast
is perfectly clean. If every male was processed in one batch and every female in
another, a sex effect and a batch effect are the same contrast; the audit reports
that as an error while still passing the main comparison. A covariate merely
nested within condition — age in weeks inside aged/adult — is the definition of
the groups rather than a confound, and does not affect the technical verdict.

The audit also reports **within-mouse contrasts**: sections of the same animal
that differ in a technical factor. Same biology, differing only technically —
the cleanest measurement of a technical effect available in the study, and worth
knowing about before anything else.

The negative-control rate is reported two ways. `control_rate` counts only
designed-negative probes and codewords — stable across panel designs, so it is
the number compared between runs. `background_rate` counts unassigned and
deprecated codewords separately, because which codewords fall in those classes
depends on the panel design and the Ranger version: folding them in would let a
version difference read as a quality difference.

### Two decisions worth knowing about

**Cross-run comparisons use only the genes present in every run.** Include a gene
one run's panel lacked and its zero column is indistinguishable from genuine
non-expression, so the pseudobulk PCA separates runs by panel and reports it as a
batch effect. That artefact is avoidable, so it is avoided — `safe_gene_set.txt`
lists what survived, and `panel_retention.csv` shows what each alternative
threshold would have cost.

**Effect sizes lead, not p-values.** With 6–16 pseudobulk samples a
Kruskal-Wallis p is coarse and a variance-component model is not identifiable.
The audit reports eta-squared per PC per factor, alongside the value that factor
would reach *by arithmetic alone* given its number of levels — a 6-level factor
over 8 samples explains 71% of everything and means nothing. Only a clear excess
over that baseline is called a driver.

---

## Splitting the analysis when a covariate is confounded

If a covariate turns out to be perfectly aliased with a technical factor, the
usual response is to analyse within its levels:

```bash
xenium-lea audit --manifest manifest.csv --split-by sex --out audit_out
```

That writes `audit_out/index.html` comparing every analysis side by side, plus a
full report for each stratum and for the pooled study.

It is worth being precise about the trade, because splitting is not simply the
safe option:

| | |
|---|---|
| **Gains** | Inside a stratum the aliased factors are **constant** — no panel difference, no segmentation difference, nothing to adjust for. The **whole panel** becomes usable, not just the genes shared across designs. |
| **Costs** | Replicates. An n=4 vs n=4 comparison split in half is n=2 vs n=2, and no stratum borrows strength from the other. |
| **Forecloses** | The strata can no longer be compared. Splitting on a confounded covariate **accepts losing** that comparison rather than recovering it. |
| **Becomes** | Two independent replicates of the same question under entirely different technical conditions. An effect present in **both** is stronger evidence than one pooled result, because no shared technical artefact could produce it. |

The exit status follows the strata, since those are the analyses that will be
used. The pooled report is kept as the record of why the split was needed, and
if it carries errors the run says so explicitly rather than letting exit 0 read
as "nothing found".

---

## Getting data to the audit

Per run, the audit reads only three things — the panel, the cells, and the run
metadata:

| | Read from, in order |
|---|---|
| Panel | `cell_feature_matrix/features.tsv.gz` → `cell_feature_matrix.zarr.zip` → `cell_feature_matrix.h5` → `gene_panel.json` |
| Cells | `cells.parquet` → `cells.csv.gz` → `cells.csv` |
| Metadata | `experiment.xenium` and `metrics_summary.csv` |
| Counts (`--deep` only) | `cell_feature_matrix/matrix.mtx.gz` → `cell_feature_matrix.h5` → `cell_feature_matrix.zarr.zip` |

**Start with `metrics_summary.csv` alone.** A run directory holding nothing but
that one file — a few kB — still yields an inventory row, a segmentation call,
run-level QC and the **full separability verdict**. It states the segmentation
split outright (`segmented_cell_stain_frac`), names the custom add-on design
(`panel_design_id`), and carries covariates that appear nowhere else: section
thickness, transcript density per area, fraction of transcripts assigned. Only
add-on *gene membership* needs the bundles, so you can get the verdict before
copying anything large:

```
audit_root/runs/<region>/metrics_summary.csv
```

It never opens `transcripts.parquet`, boundary parquets, or morphology images —
the bulk of a Xenium bundle. **Tier 0** (everything except question 7, including
the verdict) is a few MB per run. `--deep` adds the count matrix and caches its
per-gene totals, so the cost is paid once.

Xenium Ranger changed both the containers and the vocabulary across versions, so
the audit takes whichever form is present and normalises it. That normalisation
is not cosmetic: a v4 run labels its RNA targets `Gene Expression` and a v6 zarr
labels them `gene`, and v6 appends a synthetic `Total transcripts` row that would
dominate every downstream number if counted as a gene. A study spanning two
Ranger versions would otherwise show a "panel difference" that is pure
nomenclature. Reading the panel from a zarr needs no zarr library — the feature
list is a plain JSON member of the archive. Only `--deep` against a zarr or `.h5`
matrix needs the optional extras (`pip install -e ".[zarr]"` / `".[h5]"`).

---

## Relationship to `xenium-spatial`

[`xenium-spatial`](https://github.com/AlexanderJais/xenium-spatial) is the
analysis pipeline — ROI selection, pseudobulk PCA, Leiden clustering,
composition, DGE, spatial niches. `xenium_LEA` is standalone and does not import
it; run the audit first, then analyse.

Several things the audit surfaces are not visible from inside that pipeline:

- Its loader **discards negative-control and blank codewords** at load. Those are
  the cleanest *panel-independent* measure of run quality — exactly what a study
  with differing add-on panels needs — so the audit keeps and reports them.
- It reads `experiment.xenium` into `uns` and **never parses it**. Panel
  identity, software version and the declared segmentation settings all live in
  there.
- `cells.parquet` QC columns land in `.obs` and are **never read**.
- **There is no cell QC filtering anywhere**, so near-empty segmentation
  artefacts are normalised up into noise. The audit reports what a filter would
  remove per run — and if that fraction differs sharply across runs, the filter
  is itself a batch effect.
- Its panel harmonisation records `var['zero_filled_any']`, but **nothing
  downstream consumes it**; and base-panel genes are hard-coded as never
  zero-filled, so a missing base gene is filled silently. The audit checks base
  completeness independently and treats a gap as an error.
- `replicate` defaults to `slide_id`, which **pseudoreplicates** when a mouse
  contributes several sections.
- Harmony defaults **on** for multi-slide runs with `batch` defaulting to
  `slide_id` — i.e. correcting across conditions by default.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

Tests build structurally faithful synthetic Xenium bundles on disk (real MTX
layout, control rows in `features.tsv.gz`, both segmentation generations), so no
real data is needed. `tests/test_design.py` pins each separability verdict
against a design whose right answer is known by construction.

## Licence

MIT.
