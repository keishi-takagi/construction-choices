# Construction Choices in a Reference-Based Topological Regime Classifier — Reproduction Code

**Construction Choices in a Reference-Based Topological Regime Classifier:**
*Normalisation, Reference Selection, and What the Wasserstein Distance Measures*

Keishi Takagi, Independent Researcher

> *Working Paper — Empirical Evidence from HYG Daily Data, 2007–2026*

---

## Overview

This repository contains the full reproduction script for all tables
and figures in the paper.

The paper examines the construction choices behind the HYG GFC
Wasserstein topological regime classifier introduced in Takagi (2026a)
— *Topological Memory of the GFC Nadir* — rather than its predictive
content. The classifier is built in four steps: an H1 persistence
diagram of each 60-day window, a Wasserstein-2 distance to a fixed
reference diagram, a 252-day rolling Z-score of that distance, and a
threshold at `z <= -1.5`. The script varies each choice that the source
paper fixed and reports what changes.

**(1) Normalisation.** Two constructions of the Wasserstein-2 distance
between persistence diagrams are implemented side by side:

| Construction | Measure on the augmented diagrams | Paper | Used in |
|---|---|---|---|
| `published` | uniform probability measures | eq. (2) | Takagi (2026a) |
| `counting`  | counting measure | eq. (1) | stability literature |

The two differ by a factor of `n^{-1/2}`, `n = |D1| + |D2|`.

**(2) Reference selection.** The reference diagram is taken from three
crisis nadirs — GFC (2009-03-09), European debt crisis (2011-10-03) and
COVID (2020-03-23) — in both HYG and JNK.

**(3) What the distance measures.** The squared distance is compared
with `S_t`, the sum of squared persistences of the *current* diagram
alone, which uses no reference information. Because the rolling
Z-score removes any additive constant exactly, the reference-dependent
part of the squared distance does not survive into the thresholded
quantity. Classifiers that use **no reference diagram at all** (maximum
persistence, total persistence, H1 loop count) are built on the same
panel and compared against the reference-based ones.

**(4) What a low `S_t` describes.** `S_t` is characterised against the
path efficiency of the 60-day window (net displacement over total
distance travelled), against realised volatility, and against the
within-window position of the minimum price; the out-of-sample
classified days are stratified by S&P 500 drawdown from the 252-day
peak.

**(5) Scope check.** The same composition — distance to a historical
reference, standardised over a trailing window, then thresholded — is
applied to a Mahalanobis turbulence index (Kritzman and Li 2010), which
contains no topology, to show how much of the thresholded quantity
survives discarding the reference.

**Verification.** The script checks that the published
(uniform-probability) implementation reproduces Takagi (2026a, SSRN
7057698) Table 1 cell for cell and prints `PASS` or `FAIL` with a
diagnosis (see `README_data.md`).

**The exercise is descriptive throughout.** It characterises how a
topological classifier's output depends on its construction; it does
not constitute a forecast, a causal claim, an investment
recommendation, or a trading rule.

---

## Repository Structure

```
.
├── reproduce_construction_choices.py  # Main reproduction script
├── requirements.txt                   # Python dependencies
├── README.md
├── README_data.md                     # Data sourcing instructions
├── LICENSE
├── .gitignore
├── data/                              # Input CSVs (not included; see README_data.md)
├── .cache/                            # PD / distance caches (created on run)
└── out/                               # Output directory (created on run)
    ├── tables/
    │   ├── all_tables.txt             # Full stdout mirror
    │   └── table*.csv                 # One CSV per table / sub-table
    └── figures/
        └── fig*.pdf, fig*.png         # 8 figures, each in PDF and PNG
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

The script requires `ripser` (Vietoris–Rips persistent homology) and
`pot` (Python Optimal Transport, for the Wasserstein-2 distance), plus
`numpy` and `scipy`. `matplotlib` is needed only for figures.

### 2. Prepare data

Place the three required CSV files in `data/` — see `README_data.md`
for sourcing (Yahoo Finance) and the exact expected format.

### 3. Run — full set (all tables + all 8 figures)

```bash
python reproduce_construction_choices.py \
    --hyg   data/longterm_hyg.csv \
    --jnk   data/longterm_jnk.csv \
    --sp500 data/longterm_sp500.csv \
    --outdir out/
```

With the files in `data/`, the paths can be omitted — they are the
defaults.

### 4. Run — tables only (faster)

```bash
python reproduce_construction_choices.py --no-figures
```

The first run computes a persistence diagram for every daily 60-day
window of HYG and JNK and six Wasserstein distance series (three
references × two normalisations), then runs the circular-shift
surrogate tests. Diagrams and distances are cached under `.cache/`, so
subsequent runs on the same input skip that step.

---

## Options

| Option | Default | Description |
|---|---|---|
| `--hyg` | `data/longterm_hyg.csv` | HYG closing prices |
| `--jnk` | `data/longterm_jnk.csv` | JNK closing prices |
| `--sp500` | `data/longterm_sp500.csv` | S&P 500 closing prices |
| `--outdir` | `out` | Output directory for tables and figures |
| `--end-date` | `2026-05-15` | Truncate all series at this date. The default is the data endpoint of Takagi (2026a); `none` uses every observation |
| `--normalisation` | `both` | `published`, `counting`, or `both` |
| `--surrogates` | `2000` | Circular-shift surrogate draws |
| `--seed` | `20260819` | RNG seed |
| `--no-figures` | off | Skip figure generation |
| `--cache` | `.cache` | Directory for persistence-diagram / distance caches |

Fixed parameters, following Takagi (2026a): Takens window `W = 60`,
dimension `D = 3`, delay `τ = 3`; Vietoris–Rips filtration threshold at
the 70th percentile of pairwise distances; rolling Z-score window 252
trading days; classifier threshold `z_t <= -1.5`; forward-return
horizon 20 trading days; in-sample through 2019-12-31, out-of-sample
from 2020-01-01.

---

## Outputs

### Tables (printed to stdout, mirrored to `out/tables/all_tables.txt`)

Tables are printed in the order the script computes them (Table 16 is
printed before Tables 13 and 15).

| # | Content | CSV |
|---|---------|-----|
| 1 | Analysis panel and measured unconditional reference (base rates on the reachable panel); days excluded as unreachable | `table1_base_rates.csv`, `table1b_reachability.csv` |
| 2 | Dependence structure of the overlapping outcome variable: ACF, zero crossing, Bartlett-truncated variance-inflation factor, effective N | `table2_dependence.csv` |
| 3 | Normalisation sensitivity (GFC reference), with the verification against Takagi (2026a) Table 1 | `table3_normalisation.csv` |
| 4 | Correlation of the squared distance with the current diagram alone (`S_t`) | `table4_distance_content.csv` |
| 5 | Structure of the reference diagrams (HYG and JNK, three nadirs) | `table5_reference_structure.csv` |
| 6 | Reference selection sensitivity (HYG price series) | `table6_reference_selection.csv` |
| 7 | Classifiers using no reference diagram | `table7_reference_free.csv` |
| 8 | Jaccard index between classified-day sets | `table8_jaccard.csv` |
| 9 | Circular-shift surrogate test | `table9_shift_null.csv` |
| 10 | Joint agreement of the configurations under a common circular shift; threshold-free order-statistic version | `table10_joint_grid.csv`, `table10b_joint_threshold_free.csv` |
| 11 | Structural statistics of classified days (GFC reference) | `table11_structure.csv`, `table11b_distance_structure.csv` |
| 12 | The within-window minimum-price position (`min_pos`) | `table12_minpos_deciles.csv`, `table12b_minpos_classifier.csv` |
| 13 | Behaviour of the rolling Z-score denominator | `table13_zscore_denominator.csv` |
| 14 | Non-topological alternatives: realised volatility | `table14_volatility_correlation.csv`, `table14b_volatility_classifier.csv`, `table14c_volatility_overlap.csv` |
| 15 | Threshold marginality: days within a given distance of the threshold | `table15_threshold_marginality.csv` |
| 16 | What a low `S_t` describes: path efficiency of the 60-day window | `table16_path_efficiency_deciles.csv`, `table16b_path_efficiency_classified.csv` |
| 17 | Out-of-sample classified days by S&P 500 drawdown from the 252-day peak | `table17_drawdown_strata.csv`, `table17b_panel_cumulative_return.csv` |
| 18 | Activation clustering (episodes separated by more than 20 trading days), and the outcome variable | `table18_activation_episodes.csv`, `table18b_hyg_own_outcome.csv` |
| 19 | The same composition applied to a Mahalanobis turbulence index | `table19_turbulence_analogue.csv` |

### Figures (saved to `out/figures/`, PDF and PNG)

| Output file | Content |
|-------------|---------|
| `fig1_distance_content` | Squared distance against the current diagram's own squared persistence |
| `fig2_structure` | Structural statistics of classified days against the full sample (requires `--normalisation both`) |
| `fig3_reference_sensitivity` | In-sample and out-of-sample lift by configuration |
| `fig4_reference_free` | Reference-free classifiers and day-set overlap |
| `fig5_shift_null` | Circular-shift null distributions with the observed statistic marked |
| `fig6_minpos` | `min_pos` decile profile against diagram complexity |
| `fig7_path_efficiency` | What a low `S_t` describes — path efficiency of the 60-day window |
| `fig8_drawdown_strata` | Where the OOS classified days sit in the drawdown distribution |

---

## Methodology

```
Persistence Diagram (PD) construction — identical to Takagi (2026a):
  1. For each trading day t, take the 60-day closing-price window (W = 60).
  2. Apply Takens delay embedding (D = 3, tau = 3), yielding
     W - (D-1)tau = 54 points in R^3.
  3. Centre the point cloud coordinate-wise and scale by the
     full-cloud scalar standard deviation.
  4. Compute Vietoris-Rips persistent homology via Ripser, with the
     filtration threshold at the 70th percentile of pairwise distances.
  5. Retain the finite H1 (loop) persistence diagram.

Reference PDs: the H1 diagram of the 60-day window ending on each
nadir (GFC 2009-03-09, European 2011-10-03, COVID 2020-03-23),
in HYG and in JNK.

Distance (diagonal augmentation, POT):
  counting=True   eq. (1)  counting measure
  counting=False  eq. (2)  uniform probability measures (Takagi 2026a)

Classifier:
  z_t = (w_t - mean_252(w)) / std_252(w)
  classified day: z_t <= -1.5
  Reference-free variants replace w_t with maximum persistence,
  total persistence, or H1 loop count of D_t.

Panel:
  The base rate is measured on the reachable panel -- days after the
  60-day Takens warm-up and the 252-day Z-score warm-up -- on which
  every construction, reference-based or reference-free, is computable.

Outcome variable:
  r_{t,20} = p^SPX_{t+20} / p^SPX_t - 1   (20-day forward S&P 500 return)
  Conditional frequencies are reported against the measured base rate,
  with effective sample sizes from the Bartlett-truncated
  variance-inflation factor of the overlapping outcome.

Inference:
  Circular-shift surrogates of the Z-score series, preserving its own
  serial correlation; for the joint test, a common shift offset is
  applied to every configuration in each draw, preserving the
  cross-configuration correlation structure.
```

---

## Limitations

- **Single outcome, single asset class** — The outcome variable is the
  20-day forward S&P 500 return throughout; the classifier is built on
  HYG, with JNK used for reference diagrams and the turbulence analogue.
- **Overlapping outcome** — Effective sample sizes are reported, but
  classified days cluster into a smaller number of activation episodes
  (Table 18), which a day count treats as independent draws.
- **Threshold marginality** — A small number of days sit close enough
  to the threshold that data-vintage differences in the fourth decimal
  can move them (Table 15, `README_data.md`).
- **Turbulence analogue** — Table 19 is a scope check on the
  composition, not a finding about turbulence: its outcome variable,
  base rate and dependence structure are outside this paper.
- **No causal claim / no tradability claim** — The paper does not
  propose, simulate, or recommend any exposure rule conditioned on the
  classifier.

---

## Requirements

- Python ≥ 3.8
- numpy, scipy (required)
- ripser, pot (required — persistent homology and optimal transport)
- matplotlib (optional — not needed with `--no-figures`)

See `requirements.txt`.

---

## Citation

```bibtex
@misc{takagi2026construction,
  author       = {Takagi, Keishi},
  title        = {Construction Choices in a Reference-Based Topological
                  Regime Classifier: Normalisation, Reference Selection,
                  and What the Wasserstein Distance Measures},
  year         = {2026},
  howpublished = {Working Paper}
}
```

---

## Related Work

- **Takagi (2026a)** (SSRN 7057698): *Topological Memory of the GFC
  Nadir: A Regime Classifier for HYG Based on Wasserstein Proximity to
  Crisis Persistence Diagrams* — introduces the classifier whose
  construction is examined in this paper.

- **Takagi (2026)**: *Base Rates and Autocorrelation-Robust Diagnostics
  for a Topological Regime Classifier* — the companion paper subjecting
  the same classifier to quasi-experimental diagnostics under
  overlapping return horizons.

---

## License

MIT License — see `LICENSE`.

---

## Disclaimer

This code and paper are provided for research purposes only. The
findings are descriptive characterisations of how a topological
classifier depends on its construction, not forecasts of future returns
or recommendations. Nothing in this repository constitutes financial
advice. Past empirical regularities described herein do not guarantee,
and should not be construed as implying, future results.
