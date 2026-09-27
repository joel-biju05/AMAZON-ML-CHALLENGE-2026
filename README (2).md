# Amazon ML Challenge 2026 - Business Entity Resolution Pipeline

This repository contains an end-to-end, vectorized Machine Learning pipeline for large-scale Business Entity Resolution across three disparate data sources (`Source 1`, `Source 2`, and `Source 3`). 

The pipeline handles data loading, string normalization, candidate blocking, feature extraction, supervised classification, validation tuning for $F_{0.5}$, and submission formatting in full compliance with the Amazon ML Challenge 2026 evaluation criteria and submission specifications.

---

## Table of Contents

- [Overview & Architecture](#overview--architecture)
- [Pipeline Stages](#pipeline-stages)
- [Directory Structure](#directory-structure)
- [Requirements & Dependencies](#requirements--dependencies)
- [Setup & Environment Preparation](#setup--environment-preparation)
- [Execution Instructions](#execution-instructions)
- [Reproducibility & Hardware Adaptations](#reproducibility--hardware-adaptations)
- [Outputs & Verification](#outputs--verification)

---

## Overview & Architecture

In large-scale commercial platforms, entity identity data arrives from multiple independent sources without global shared primary keys. This pipeline resolves entity records from **Source 2** and **Source 3** back to reference records in **Source 1**.

### Key Architectural Highlights

1. **Scalable Vectorized Blocking:** Uses $N$-way string signatures and bounded candidate joins via Pandas to reduce $O(N \cdot M)$ space complexity while protecting memory.
2. **Memory-Aware Dynamic Scaling:** Monitors system memory (`psutil`) and automatically tunes bucket caps, candidate limits, and batch sizes to prevent out-of-memory errors on modest hardware.
3. **C++ Accelerated Feature Extraction:** Harnesses C++-backed string similarity calculations (`rapidfuzz.process.cdist`) across normalized fields.
4. **$F_{0.5}$-Optimized Classification:** Trains a LightGBM (or scikit-learn `HistGradientBoostingClassifier` fallback) and tunes decision thresholds explicitly to optimize macro $F_{0.5}$, heavily penalizing false merges while giving full credit for singletons.
5. **Strict Challenge Compliance:** Outputs properly formatted Tab-Separated Value (`.tsv`) files matching all challenge constraints, including support for singletons, empty matches, non-duplicate IDs, and open-set country attributes (US, India, France).

---

## Pipeline Stages

```
┌─────────────────────────────────────────────────────────┐
│ Stage 1: Zip & Column Auto-Detection                   │
│   - Load train/test data directly from compressed Zip  │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│ Stage 2: String & Character Normalization               │
│   - Unicode NFKC, entity synonyms, digit extraction     │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│ Stage 3: Multi-Key Bounded Candidate Blocking           │
│   - Multi-field exact/prefix/signature joins            │
│   - Top-K capping per S1 entity                        │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│ Stage 4: High-Performance Feature Engineering          │
│   - RapidFuzz ratios, token sort/set, digit & country   │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│ Stage 5: Entity-Split Model Training & Downsampling     │
│   - Entity-level split preventing target leakage       │
│   - Gradient Boosted Decision Trees (LightGBM/HistGB)   │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│ Stage 6: Validation & F_0.5 Threshold Tuning            │
│   - Grid search over decision boundaries for F_0.5      │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│ Stage 7: Inference & Required Deliverable Generation    │
│   - Produces candidate_pairs.tsv & matching_results.tsv │
└─────────────────────────────────────────────────────────┘
```

---

## Directory Structure

The repository structure inside the final submission package (`<team_name>_submission.zip`) must follow this layout:

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv            # Final matches (uploaded to Portal)
│   └── candidate_pairs.tsv             # Final blocking set fed to ML model
└── business_entity_resolution/
    ├── README.md                       # This instruction file
    ├── requirements.txt                # Pinned dependency file
    ├── Documentation_template.md       # Methodology documentation
    └── src/
        └── entity_resolution_pipeline_FINAL_SAFE.py  # Main pipeline execution script
```

---

## Requirements & Dependencies

The pipeline requires **Python 3.9+** and standard data science libraries.

### Core Libraries
- `numpy` ($\ge 1.21.0$)
- `pandas` ($\ge 1.3.0$)
- `scikit-learn` ($\ge 1.0.0$)
- `rapidfuzz` ($\ge 3.0.0$)

### Optional (Recommended)
- `lightgbm` ($\ge 3.3.0$) — Preferred model backend for faster training and superior accuracy.
- `psutil` ($\ge 5.8.0$) — Enables real-time RAM monitoring and automatic safety scaling.

A sample `requirements.txt`:
```text
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
rapidfuzz>=3.0.0
lightgbm>=3.3.0
psutil>=5.8.0
```

---

## Setup & Environment Preparation

1. **Clone/Unpack the Repository Code:**
   Ensure the source file `entity_resolution_pipeline_FINAL_SAFE.py` resides inside `business_entity_resolution/src/`.

2. **Create a Virtual Environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r business_entity_resolution/requirements.txt
   ```

4. **Prepare Challenge Dataset:**
   Place your challenge ZIP archive (e.g., `6ab10eb3b23ba_student_resource.zip`) in your user `Downloads` folder or update `ZIP_PATH` in the code.

---

## Execution Instructions

Run the complete pipeline end-to-end using a single command from the root directory:

```bash
python3 business_entity_resolution/src/entity_resolution_pipeline_FINAL_SAFE.py
```

### What Happens During Execution

1. **Auto-Discovery:** Automatically locates the student resource ZIP archive and identifies table columns (`entity_id`, `business_name`, `business_address`, `country`).
2. **RAM Diagnostic & Auto-Tuning:** Detects available physical memory and automatically adjusts internal batch limits (`BUCKET_CAP`, `MAX_CANDIDATES_PER_S1`, `FEATURE_BATCH`).
3. **Training & Threshold Optimization:** Executes train/validation splits, extracts pairwise string features, fits the Gradient Boosted Classifier, and finds the exact decision boundary maximizing macro $F_{0.5}$.
4. **Inference & Output Generation:** Runs candidate generation and predictions on test data, writing all finalized deliverables to `~/amazon_ml_entity_resolution_output/` (or the configured `OUTPUT_DIR`).

---

## Reproducibility & Hardware Adaptations

To guarantee identical results across systems, a fixed seed (`RANDOM_SEED = 42`) is set for model training, data splitting, and negative sampling.

### Configurable Hyperparameters

You can adjust parameters at the top of `entity_resolution_pipeline_FINAL_SAFE.py`:

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `RANDOM_SEED` | `42` | Global random seed for reproducible splits and training. |
| `BUCKET_CAP` | `100` | Max rows kept per blocking key side during merging. |
| `MAX_PAIRS_PER_KEY` | `5000` | Max total pairs permitted for a single key before dropping. |
| `MAX_CANDIDATES_PER_S1` | `8` | Upper bound of candidates retained per Source-1 entity. |
| `NEGATIVE_PER_POSITIVE` | `8` | Sampling ratio of negative pairs to positive pairs for training. |
| `FEATURE_BATCH` | `50000` | Batch size for memory-bounded rapidfuzz feature calculation. |

---

## Outputs & Verification

Upon execution completion, the script generates the following files in the designated output directory:

| File Name | Description | Required Packaging Location |
| :--- | :--- | :--- |
| `matching_results.tsv` | Final predictions containing predicted S2/S3 entity match lists for every S1 entity. | `output/matching_results.tsv` |
| `candidate_pairs.tsv` | Candidate pool generated by the blocking stage prior to final ML classification. | `output/candidate_pairs.tsv` |
| `validation_metrics.csv` | Honest macro $F_{0.5}$, Precision, and Recall scores measured on held-out validation entities. | Reference / Internal |
| `feature_importance.csv` | Feature importance ranking breakdown. | Reference / Internal |

### Local Submission Validation

Before submitting your package to the evaluation portal, validate the generated outputs using the provided utility script:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

If formatting rules are satisfied, the script will return `PASS` with exit code `0`.