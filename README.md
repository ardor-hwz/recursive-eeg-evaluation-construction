# Evaluation Construction and Conditional Interpretation in Recursive EEG-Based PERCLOS-Related Drowsiness Tracking

## Overview

Analysis code, a frozen perturbation protocol, and aggregate result files supporting analyses reported in:

**Evaluation Construction and Conditional Interpretation in Recursive EEG-Based PERCLOS-Related Drowsiness Tracking**

## Repository contents

```text
README.md
requirements.txt
src/
  primary/
  scope_checks/
configs/
  primary/
scripts/
  construction_contrast/
  selection_audit/
  scope_checks/
    external/
results/
```

- `src/`: original scientific implementation modules.
- `configs/`: the original frozen M1 perturbation protocol.
- `scripts/`: original comparator, perturbation, selection-audit and external-analysis scripts.
- `results/`: saved aggregate scientific outputs.
- `requirements.txt`: the historical Python dependency declaration.
- `.gitignore`: exclusions for datasets, caches and credentials.

The supplied source files preserve their original contents and module imports. Files containing fixed machine-specific absolute paths were withheld. Individual analyses depend on their original input artifacts and module bindings; this archive does not supply an automated raw-data-to-all-results workflow.

## Data availability

Original SEED-VIG and Cao sustained-attention driving data are not included. Obtain data from their original providers and follow the applicable access and redistribution terms.

- SEED-VIG: [BCMI Laboratory dataset portal](https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html).
- Cao driving data: [original dataset description](https://doi.org/10.1038/s41597-019-0027-4).

Raw EEG, features, labels, individual trajectories and model checkpoints are not distributed in this package.

## Analysis scope

| Included material | Location |
|---|---|
| Primary SEED-VIG loading/alignment, features, nested selection, grouped stacking, recursive updater, constant comparator, perturbation endpoints and inference | `src/primary/` |
| Comparator and perturbation-construction analyses, including W01/W02/W04 | `scripts/construction_contrast/` |
| Selection-conditioned claim-retention analysis using saved inputs | `scripts/selection_audit/` |
| W06 same-source adaptive-Kalman updater kernels | `src/scope_checks/source_adaptive_kalman/` |
| External parsing, feature/QC processing, nested proxy fitting and participant-level estimation | `src/scope_checks/external/` and `scripts/scope_checks/external/` |
| Original frozen M1 perturbation definition | `configs/primary/M1_LITE_FROZEN_PROTOCOL.json` |

The supplied external code covers the listed components. Its machine-bound root/replay modules and several protocol locks are not included. Scientific results from those analyses are preserved in the aggregate tables below.

## Results

All CSV files listed below are located in `results/`.

| Files | Saved output |
|---|---|
| `P0_RMSE_CONFIRMATORY_FAMILY.csv` | Primary predictive RMSE contrasts |
| `W01_result_table.csv`, `W01_CANONICAL_COMPARATOR_ESTIMATES.csv` | Comparator-construction summaries and manuscript-oriented comparator contrasts |
| `W02_population_summary.csv` | Perturbation-response components and interactions |
| `W04_POPULATION_SUMMARY.csv`, `W04_POPULATION_CELLS.csv` | Perturbation-construction contrast and descriptive components |
| `W06_COMPARATOR_SUMMARY.csv`, `W06_M1_COMPONENT_SUMMARY.csv`, `W06_M1_CONSTRUCTION_SUMMARY.csv` | Adaptive-Kalman comparator, component and construction summaries |
| `W06_HEURISTIC_COMPARATOR_SUMMARY.csv`, `W06_HEURISTIC_MATCHED_SUMMARY.csv` | Corresponding heuristic-updater and matched summaries |
| `EXPLORATORY_ESTIMATES.csv`, `POPULATION_COMPONENT_AREAS.csv` | W09E external exploratory estimates and component areas |

The primary source analyses use 21 prefix-defined groups across 23 recordings. W09E uses a separate support-qualified exploratory setting with 19 participants. The original planned full-cohort external endpoint remains non-evaluable.

Historical W01 uses `K=C0-Cmean`; its manuscript-oriented comparator contrast is `Delta=Cmean-C0=-K`. The existing canonical W01 table omits individual group-ID lists and local path columns. All supplied result files retain the contents of the input publication package.

## Figures

Figure 1 is an author-created conceptual schematic and is not generated programmatically. No automatic regeneration of all manuscript figures is claimed.

## Environment

Use Python 3.11 or later for the supplied scientific scripts. `requirements.txt` is copied from the original project's dependency declaration; it includes NumPy, SciPy, pandas, scikit-learn, h5py, PyYAML, pyarrow and CPU PyTorch. It records historical requirements rather than a newly validated environment for all branches.

```bash
python -m pip install -r requirements.txt
```
