# ADMET-Stack — code

Code for *ADMET-Stack: Quantitative Guidance on How Much Representational Complexity an ADMET
Endpoint Actually Requires* (Omid Mahmoudi and Mi-Hyun Kim).

Two folders:

- **`pipeline/`** — the code that produced the models: structure preparation, feature building,
  the stacked ensemble, and the external evaluations.
- **`analysis/`** — the statistical analyses added during revision. These are the scripts behind
  the new supplementary tables. Most of them run on their own, without the large run artifacts.

Everything is plain Python. Install with `pip install -r requirements.txt`, or build the conda
environment from `environment.yml`.

---

## pipeline/

Run in order. Each script needs a data root — the folder holding the TDC datasets — which you set
in `data_root.txt` (copy `data_root.txt.example` and edit it).

| Script | What it does |
|---|---|
| `01_canonicalize_and_similarity_filter.py` | Parses input SMILES, converts them to canonical SMILES, drops anything RDKit cannot read, removes exact duplicates, then computes Morgan fingerprints (radius 2, 2,048 bits) and the maximum Tanimoto similarity of each external compound to the matching TDC endpoint. Compounds above 0.8 are removed. This is the structure-standardisation and overlap-removal step described in Section 2.8.1. |
| `02_build_features_tdc_benchmark.py` | Builds the feature matrices for the 22 TDC endpoints: Morgan, Avalon and ErG fingerprints, MACCS keys, 217 RDKit descriptors, and the four SELFIES-derived blocks. Writes one matrix per endpoint per configuration. |
| `03_build_features_external.py` | The same featurisation applied to the external datasets. |
| `04_stacking_pipeline.py` | Fits the stacked ensemble (Extra Trees, Random Forest, LightGBM, XGBoost, with a logistic or ridge meta-learner) for one endpoint and one configuration, over five random seeds, and writes the test predictions. |
| `05_hyperparameter_search.py` | The Optuna search reported in Table S25. This was a separate auxiliary study: the 54-configuration benchmark itself used library defaults, so this script is not needed to reproduce the benchmark. |
| `06_evaluate_external_datasets.py` | Scores the stored models on the external datasets and writes accuracy, AUROC, AUPRC, prevalence, majority-class rate, balanced accuracy, MCC and bootstrap intervals. It scores **every** retained run for the endpoint and reports the mean ± SD across them, because that is the basis Table S31 reports; bootstrap intervals average the per-run metric inside each replicate, as the paper describes. Where only one run was retained — the AutoML rows — the script says so and the SD is undefined. |
| `07_build_cross_isoform_probe_set.py` | Assembles the CYP2C19 probe set described in Section 2.8.3: external CYP2C19 substrates as positives, non-substrates from the held-out TDC CYP2C9 test partition as negatives, filtered to Tanimoto < 0.3 against the positives. |

`pipeline/src/` holds the two shared modules: `feature_extractor.py` (all representation blocks)
and `admet_config.py` (endpoint names, metrics, task types, and the five learner seeds).

A note on the reporting basis, since it is easy to get wrong. Every number in the paper that carries
a `±` is a mean over the five learner seeds, not the result of one run. Script `04` writes one
prediction file per seed and script `06` averages across them. If you score a single seed you will
get values close to, but not equal to, the published ones.

---

## analysis/

These scripts answer the specific methodological questions raised in review. Each one prints its
result to the terminal; nothing is overwritten.

Scripts marked **self-contained** need only the files in `data/`, which are included here:

- `data/si_tables.json` — the full benchmark grid: 54 configurations × 22 endpoints, each with the
  official TDC metric value and its across-seed standard deviation. This is the same data as
  Supplementary Tables S2–S23.
- `data/honest_selection.json` — the per-endpoint results of the split-sample selection runs.
- `data/config_dimensionalities.csv` — feature counts for every configuration and endpoint.
- `data/external_evaluation.csv` — the 28 external evaluation sets with their compound counts and
  reported accuracy, AUROC and AUPRC, exported verbatim from Table S31.

| Script | What it does | Runs out of the box? |
|---|---|---|
| `a01_verify_retention.py` | Recomputes the 94.3% retention figure from the benchmark grid, as a check on the published number. | self-contained |
| `a02_retention_sensitivity.py` | Recomputes retention five ways — arithmetic mean, median, geometric mean, 20% trimmed mean, and an AUROC chance-adjusted version — plus a non-ratio summary based on normalised regret. Shows the descriptor-only value moves between 92.9% and 96.4%. **Table S27.** | self-contained |
| `a03_winners_curse_mcda.py` | Quantifies the winner's-curse bias in the retention reference, which is the best of 54 noisy estimates per endpoint, and reproduces the MCDA block ranking. | self-contained |
| `a04_block_cluster_bootstrap.py` | Bootstrap over endpoints rather than over configurations, because configurations are reused across comparisons and are not independent. Endpoints are the sampling unit. **Table S26.** | self-contained |
| `a05_block_mixed_model.py` | Linear mixed-effects model with a random intercept per endpoint, fitted to all 48 core configurations across all 22 endpoints. A second way of handling the same dependence. **Table S26.** | self-contained |
| `a06_block_mixed_model_robustness.py` | Repeats the mixed-effects fit under alternative specifications to check the block estimates are stable. | self-contained |
| `a07_fixed_config_reanalysis.py` | Repeats the baseline comparison using a single fixed configuration across all 22 endpoints, so that no per-endpoint selection occurs, with Holm-corrected p-values and rank-biserial effect sizes. **Table S29.** | self-contained |
| `a08_split_sample_summary.py` | Summarises the split-sample results: how much performance is inflated by choosing the configuration on the same test set it is scored on. **Table S30.** | self-contained |
| `a09_classification_difference_CI.py` | Bootstrap confidence intervals for the mean ADMET-Stack minus baseline difference, with equivalence tests against a ±5% margin. This replaces the earlier claim that classification performance was indistinguishable from the baselines. **Table S38.** | self-contained |
| `a10_rdkit_descriptor_composition.py` | Counts how many of the 217 RDKit descriptors actually encode structure: 85 SMARTS-based fragment counts plus 3 Morgan-fingerprint densities, so 88 of 217. **Table S1.** | self-contained |
| `a11_external_prevalence_audit.py` | For every external evaluation set, computes positive-class prevalence, the majority-class rate and the AUPRC lift over prevalence, and flags the rows where accuracy falls below the majority-class rate, where AUROC is below chance, or where fewer than 100 compounds were evaluated. Reads the Table S31 grid from `data/external_evaluation.csv`, so it cannot drift from the published table. **Table S31.** | self-contained |
| `a12_bootstrap_test_CIs.py` | Bootstrap confidence intervals over test compounds, 2,000 replicates, with the metric averaged across the five seeds inside each replicate. **Table S28.** | needs stored predictions |
| `a13_split_sample_selection.py` | Runs the split-sample procedure that `a08` summarises: each test set is split in half 250 times, the best configuration is chosen on one half and scored on the other. | needs stored predictions |
| `a14_config_dimensionalities.py` | Reads the stored feature matrices and records the real width of each one, which is how the discrepancies in Table S34 were found. **Table S24.** | needs stored feature matrices |
| `a15_validation_selected_configs.py` | Harvests a separate run in which the validation partition was held out instead of pooled, so configurations could be chosen on validation and then tested. **Table S44.** | needs that run's outputs |
| `a16_inhouse_ames_overlap.py` | Measures the overlap between the in-house AMES collection and TDC AMES: 2,735 of 4,497 distinct structures are exact matches, 2,189 of them in the pooled training data. **Section 2.8.4.** | needs the in-house file |
| `a17_inhouse_ames_recompute.py` | Rescores the in-house AMES predictions on the deduplicated, overlap-filtered subset of 1,390 compounds, where AUROC falls from 0.916 to 0.746. **Section 2.8.4.** | needs the in-house file |
| `a18_diagnose_below_chance.py` | Diagnostics for the two external sets that returned below-chance AUROC — checks label polarity and feature-column alignment. | needs the external files |
| `a19_make_fig_S6.py` | Draws Fig. S6: QED against the aggregate ADMET risk-favorability score over the 1,088-compound reference set. | needs the reference predictions |
| `a20_benchmark_cost.py` | Optional. Measures feature-extraction time, training time, inference time and peak memory per configuration. These numbers were not profiled during the original runs, which is why the paper makes no efficiency claim; this script is provided so the measurement can be made. | needs the data root |

For the scripts in the second group, set the path constant at the top of the file, or point
`ADMET_DATA` at a folder holding the inputs. The large run artifacts — processed features, fitted
models and raw predictions — are in the main repository under `data/`, `models/` and `results/`.

---

## Reproducing the headline numbers

```bash
cd analysis
python a01_verify_retention.py          # the 94.3% figure
python a02_retention_sensitivity.py     # and the 92.9-96.4% range behind it
python a07_fixed_config_reanalysis.py   # the baseline comparison without per-endpoint selection
python a08_split_sample_summary.py      # how much that selection was worth
```

The fourth of these is the one worth reading first. It shows that configuration selection on the
test partition inflates apparent performance by a median of 0.035 on the 13 endpoints with fewer
than 300 test compounds, and that per-endpoint selection beats a single fixed configuration on only
5 of 22 endpoints once selection and scoring are separated.
