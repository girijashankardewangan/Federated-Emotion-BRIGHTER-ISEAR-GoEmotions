# Inter-Auditor Protocol

Purpose: enable independent verification of reported fixture outcomes.

## Setup

1. Clone repo:
   git clone https://github.com/girijashankardewangan/Federated-Emotion-BRIGHTER-ISEAR-GoEmotions.git
   cd Federated-Emotion-BRIGHTER-ISEAR-GoEmotions

2. Install:
   pip install pandas numpy torch

3. DO NOT read oracle_results.json, pytorch_fixtures.json, or
   software_contract_checks.json before completing Step 2.

## Step 1 - Run fixtures

python scripts/verify_software_contracts.py
python scripts/evaluate_contract_oracles.py
python scripts/evaluate_pytorch_fixtures.py

## Step 2 - Classify outcomes independently

SC1 NumPy (3): early_max, later_max, tied_max
SC1 PyTorch (3): early_max, later_max, tied_max
SC2 (2): early_best, later_best
SC3 (4): isear_all_codes, brighter_missing_col, brighter_nonbinary, valid_schema_control

## Step 3 - Compare

Open oracle_results.json and pytorch_fixtures.json.
Report: Agreed: __ / 12

## Step 4 - Source-line check

Locate in train.py:
- Line assigning best_state = model.state_dict() (expected ~209)
- train_federated signature (expected ~235); is val_df used?

## Step 5 - Report

Submit signed note with agreement count, disagreements, line numbers,
environment, timestamp.

## Expected results (do not read before Step 2)

Original passes: 1/12 (SC3 valid_schema_control only)
Repair passes:   12/12
