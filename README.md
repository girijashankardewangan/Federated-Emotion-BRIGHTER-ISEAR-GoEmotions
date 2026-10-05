# Software Contracts and Experimental Validity in Machine-Learning Pipelines: An Embedded Case Study of Centralized and Federated Training

This repository contains the code, results, and audit artifacts for a software-audit case study of centralized and federated DistilBERT training for multi-label emotion classification.

## Overview

Released machine-learning software can expose training logic without documenting how each reported run was produced. This study audits one centralized and federated DistilBERT codebase and 80 released emotion-classification runs across three task settings:

- BRIGHTER English (10 seeds per method)
- Filtered ISEAR (5 seeds per method)
- Filtered GoEmotions (5 seeds per method)

Four training configurations are compared:

| ID | Configuration | Description |
|----|---------------|-------------|
| C1 | Centralized | Standard centralized DistilBERT training |
| F1 | FedAvg | Federated averaging without update perturbation |
| F2 | FedAvg + Clipping | FedAvg with completed-update clipping (C = 1.0) |
| F3 | FedAvg + Clipping + Noise | FedAvg with clipping and Gaussian noise (sigma = 1.1) |

## Repository Contents

- README.md : This file
- train.py : Released/historical training source audited in the study
- results.csv : 60 BRIGHTER/ISEAR run results
- goemotions_results.csv : 20 GoEmotions run results
- progress.json : Checkpoint progress for resuming
- brighter_summary.csv : BRIGHTER mean +/- SD
- isear_summary.csv : ISEAR mean +/- SD
- figures/ : Publication figures (PNG + PDF, 300 DPI)
- data/repository_analysis/paired_descriptive.csv : Descriptive seed-paired differences across the three reported metrics; 60 values

## Key Results

### BRIGHTER English (10 seeds per method)

| Method | Macro F1 | Micro F1 | Exact Match |
|--------|----------|----------|-------------|
| C1 | 0.1343 +/- 0.0264 | 0.4077 +/- 0.0988 | 0.1245 +/- 0.0037 |
| F1 | 0.1317 +/- 0.0403 | 0.3498 +/- 0.1389 | 0.1145 +/- 0.0256 |
| F2 | 0.1317 +/- 0.0403 | 0.3498 +/- 0.1389 | 0.1145 +/- 0.0256 |
| F3 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.1055 +/- 0.0000 |

### Filtered ISEAR (5 seeds per method)

| Method | Macro F1 | Micro F1 | Exact Match |
|--------|----------|----------|-------------|
| C1 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| F1 | 0.0167 +/- 0.0228 | 0.0322 +/- 0.0477 | 0.0205 +/- 0.0322 |
| F2 | 0.0167 +/- 0.0228 | 0.0322 +/- 0.0477 | 0.0205 +/- 0.0322 |
| F3 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |

### Filtered GoEmotions (5 seeds per method)

| Method | Macro F1 | Micro F1 | Exact Match |
|--------|----------|----------|-------------|
| C1 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| F1 | 0.0011 +/- 0.0015 | 0.0011 +/- 0.0015 | 0.0006 +/- 0.0008 |
| F2 | 0.0011 +/- 0.0015 | 0.0011 +/- 0.0015 | 0.0006 +/- 0.0008 |
| F3 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |

## Key Findings

1. F1 and F2 have identical recorded metrics at every seed on all three tasks. Equal scores do not establish whether clipping was inactive or whether it changed updates without changing thresholded scores.

2. F3 has zero macro and micro F1 across all three datasets. This is a recorded performance failure, not proof that every prediction was empty. F3's BRIGHTER exact match is 0.1055 +/- 0.0000.

3. C1 also scores zero on both filtered tasks. Federation alone cannot explain the low ISEAR and GoEmotions performance.

4. The paired comparisons are descriptive historical evidence. The matched seed differences are observed score gaps, not causal effects or population inference. Earlier significance tests are retained only as past history and are not used as evidence of equivalence or causal superiority.

## Audit Findings

The four-layer audit scheme (data/tasks, training protocol, implementation, evidence/reproduction) identified:

- Mutable centralized checkpoint state: C1 saves best_state = model.state_dict() without a deep copy. The dictionary points to tensors that later training can change.

- Unused federated validation argument: The federated routine accepts val_df but never uses it. It evaluates the final round, not the best round.

- Always-zero surprise target for ISEAR: Filtered ISEAR retains five sigmoid outputs, but surprise is always zero because ISEAR has no surprise examples.

These findings limit the interpretation of score differences but do not establish their causes.

## Reproducibility Notes

- The released scores and progress records are included, but the release does not contain the checkpoints, predictions, dataset revisions, and environment records needed to reproduce training in full.

- The analysis uses the supplied repository snapshot. File checks confirm that imported files match that snapshot, but they do not establish which code revision produced each checkpoint.

- Recomputed means and sample standard deviations agree with the released summaries at four decimal places.

## Datasets

| Dataset | Source | Task |
|---------|--------|------|
| BRIGHTER English | https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories | Multi-label, 5 emotions (joy, anger, fear, sadness, surprise) |
| Filtered ISEAR | https://github.com/bdotloh/isear_dataset | 4 retained emotions (joy, fear, anger, sadness); surprise always zero |
| Filtered GoEmotions | https://huggingface.co/datasets/google-research-datasets/go_emotions | 5-emotion subset (joy, anger, fear, sadness, surprise) |

## Verification and Reproduction Scope

- Recomputed means and sample standard deviations agree with the released summaries to four decimal places.
- Repository checks operate on the supplied repository snapshot and record relevant hashes, inputs, outcomes, and versions.
- The original historical score files and publication figures are preserved unchanged.
- Full PyTorch training was **not** rerun as part of this audit.
- Original training checkpoints, predictions, complete environments, and run-linked dataset revisions are unavailable.
- Therefore, the historical training process cannot be fully reproduced from the released evidence.
- The bounded repair checks provide local contract evidence only; they are not a replacement for end-to-end model-training validation.
- Earlier paired significance tests are historical only. The current repository retains descriptive paired differences rather than treating those tests as causal or population-level evidence.

## Bounded Repair Evaluation

The repository contains a bounded candidate repair and deterministic fixture checks for three principal software contracts.

| Contract | Fixture scope | Original passes | Candidate repair passes |
|----------|----------------|-----------------|-------------------------|
| SC1 | Selected-state preservation | 0 / 3 | 3 / 3 |
| SC2 | Validation-use wiring and best-round selection | 0 / 2 | 2 / 2 |
| SC3 | Supported task schemas and explicit rejection | 1 / 4 | 4 / 4 |
| **Total** | **9 fixtures** | **1 / 9** | **9 / 9** |

The original source therefore passes one valid-schema control and fails the eight defect-focused fixtures. The candidate repair passes all nine fixtures.

These are deterministic fixture outcomes, not additional model-training observations. They do not establish end-to-end correctness of the training pipeline. Full PyTorch training and functional training-effect tests remain outside the executed repair evidence.

## Privacy Scope

The code clips completed client updates and adds Gaussian noise. It does not implement per-example DP-SGD. No verified privacy accountant or formal (epsilon, delta) guarantee is claimed. The noise mechanism is DP-inspired only.

## Limitations

- The study does not independently rerun training.

- The four-layer audit scheme has not been validated across independent codebases.

- The three datasets are task settings within one shared implementation, not three independent software replications.

- No new learning algorithm, reproducibility metric, or formal privacy guarantee is introduced.

## Citation

If you use this code or data, please cite:

@article{dewangan2026softwarecontracts,
  title={Software Contracts and Experimental Validity in Machine-Learning Pipelines: An Embedded Case Study of Centralized and Federated Training},
  author={Dewangan, Girija Shankar and Roy, Partha and Tiwari, Rajesh},
  journal={Preprint submitted to Elsevier},
  year={2026}
}

## License

This project is released for research and reproducibility purposes. Dataset use must follow the respective dataset licenses.

## Contact

For questions, please open an issue on GitHub.
