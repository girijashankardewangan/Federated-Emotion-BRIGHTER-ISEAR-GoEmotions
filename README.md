# Federated Emotion — BRIGHTER, ISEAR, and GoEmotions

This repository accompanies the audit and reproducibility analysis
of the released DistilBERT federated-emotion implementation and its
historical experiment records.

## Historical experiment record

The repository preserves 80 historical run records:

- BRIGHTER English: 40 runs (10 seeds × 4 methods)
- Filtered ISEAR: 20 runs (5 seeds × 4 methods)
- Filtered GoEmotions: 20 runs (5 seeds × 4 methods)

Historical records are preserved in:

- `results.csv`
- `goemotions_results.csv`
- `data/repository_analysis/released_results_80.csv`

The historical results are preserved as reported. They are not
silently regenerated from the candidate repair.

## Software-contract findings

The audit identified three principal software issues in the released
implementation:

1. Centralized checkpoint selection stores `model.state_dict()`
   without deep-copying the selected state.
2. The federated routine accepts `val_df` but does not use it to
   select the best communication round; the final round is evaluated.
3. Filtered ISEAR retains the shared five-output head even though
   only joy, fear, anger, and sadness are retained.

These are software-contract findings. They should not be presented
as causal explanations for all historical model outcomes.

## Candidate repair

The historical `train.py` is intentionally preserved.

A candidate repair is provided separately under:

- `repairs/train_corrected.py`
- `repairs/train_contracts.patch`

The candidate repair is separate from the historical results and
does not constitute regenerated historical evidence.

## Contract/oracle artifacts

Repository analysis artifacts are stored under:

`data/repository_analysis/`

These artifacts distinguish source-visible findings, deterministic
contract checks, and bounded fixture/oracle evaluation.

## Figures and tables

Historical figures and tables are retained. They are not silently
replaced by candidate-repair results.

The existing figure/table inventory and hashes are recorded in:

`data/repository_analysis/figure_table_manifest.json`

## Verification

Run:

```bash
python scripts/verify_software_contracts.py
```

The verification checks the preserved historical result structure
and basic metric validity.

## Interpretation

The repository distinguishes:

- historical experimental observations;
- software-contract findings;
- bounded fixture/oracle checks; and
- candidate repairs.

Original checkpoints, predictions, and the complete historical
software environment are unavailable. Therefore contract-level
checks should not be interpreted as full end-to-end reproduction
of the historical training runs.

## Citation

After the final repository commit is created, the manuscript
reference should cite that immutable final commit rather than an
earlier development commit.
