# Supplementary Material: Software Contract Verification

Companion package for:
"Software Contracts and Experimental Validity in Machine-Learning Pipelines".

## Contents

### scripts/
- verify_software_contracts.py — SC1-SC7 checks against released archive
- build_contract_repair.py — generates deepcopy patch for SC1
- evaluate_contract_oracles.py — 9 bounded NumPy fixtures (SC1-SC3)
- evaluate_pytorch_fixtures.py — 3 additional PyTorch SC1 fixtures

### data/repository_analysis/
- software_contract_checks.json — verification report
- oracle_results.json — NumPy fixture outcomes
- pytorch_fixtures.json — PyTorch fixture outcomes
- section_5_3.md — cross-codebase audit text
- paper_revisions.md — full changelog for revised manuscript

### data/codebase_B_hf/
- run_glue.py — audited HuggingFace example (646 lines)
- audit_results.json — 4-layer audit outcomes

### repairs/
- train_contracts.patch — unified diff for SC1 deepcopy fix

## How to Reproduce

python scripts/verify_software_contracts.py
python scripts/build_contract_repair.py
python scripts/evaluate_contract_oracles.py
python scripts/evaluate_pytorch_fixtures.py

## Key Results

- SC1-SC7 verification: 80/80 rows, F1/F2 equal 60/60, F3 zero 20/20
- NumPy fixtures: original 1/9, repair 9/9
- PyTorch fixtures: original 0/3, repair 3/3
- Cross-codebase audit: HF example compliant SC1-SC3, partial SC4

## Requirements

Python 3.8+, pandas, numpy, torch
