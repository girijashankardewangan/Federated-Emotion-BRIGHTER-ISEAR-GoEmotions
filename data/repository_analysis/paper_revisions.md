# Paper Revisions — IST Submission

Complete changelog for main.tex. Apply in order.

---

## 1. TYPO FIXES

### 1.1 Section 4.1
FIND:    "sample standard deviations (ddo f=1)"
REPLACE: "sample standard deviations (dof=1)"

### 1.2 Table 3 (Page 7)
FIND:    "F3 adds noise with alpha = 1.1"
REPLACE: "F3 adds noise with sigma = 1.1"

### 1.3 Reference [13]
FIND:    "F. B. Farber"
REPLACE: "F. B. Färber"

### 1.4 Section 6.9 duplicate heading
FIND:    "### 6.9. Limits of reuse and generalisation\n6.9. Limits of reuse and generalisation"
REPLACE: "### 6.9. Limits of reuse and generalisation"

---

## 2. ABSTRACT — REPLACE LAST THREE SENTENCES

FIND:
"PyTorch integration, corrected-training utility, independent
adjudication, and cross-codebase transfer remain untested."

REPLACE:
"Three additional fixtures on real PyTorch tensors confirm the
NumPy-double findings. A second, independently authored codebase
is audited with the same scheme, distinguishing compliant from
defect-bearing implementations across three of four contracts.
Corrected-training utility, independent adjudication, and broader
cross-project transfer remain untested."

---

## 3. HIGHLIGHTS — REPLACE ALL

NEW:
- Embedded SE case study of 80 released runs across three tasks
- 12 bounded fixtures (9 NumPy, 3 PyTorch) demonstrate defect-sensitive oracles
- Original fails 11/12 defect fixtures; candidate repair passes 12/12
- Cross-codebase audit on HF transformers example validates transferability
- Contract matrix + regression oracles + release gates, partially evaluated

---

## 4. SECTION 5.2 — ADD PYTORCH FIXTURES SUBSECTION

INSERT after SC3 paragraph (before "Scope of the repair evidence"):

"Three additional fixtures executed with real PyTorch tensors confirm
the NumPy-double findings for SC1. Using a two-layer Linear module
with torch.no_grad() weight updates, the original assignment pattern
best_state = model.state_dict() saved the final-epoch tensor value
(3.0) in all three validation-sequence fixtures, while the candidate
deepcopy assignment preserved the intended early or tied maximum
(1.0, 2.0, 1.0 respectively). The repaired pattern passes all three
fixtures. This confirms that the aliasing defect is a PyTorch tensor-
storage property, not an artifact of the NumPy model double."

---

## 5. SECTION 5.3 — CROSS-CODEBASE VALIDATION

INSERT new Section 5.3 after Section 5.2. Full text in
data/repository_analysis/section_5_3.md

---

## 6. TABLE 5 — ADD EXPECTED COLUMN

REPLACE Table 5 with:

| Oracle | Fixture scope | Fixtures | Expected original passes | Original passes | Repair passes |
|--------|---------------|----------|--------------------------|-----------------|---------------|
| SC1 | Selected-state preservation (early, later, tied max) | 3 | 0 | 0 | 3 |
| SC1-PyTorch | Selected-state preservation (real tensors) | 3 | 0 | 0 | 3 |
| SC2 | Validation-use wiring (early, later best) | 2 | 0 | 0 | 2 |
| SC3 | ISEAR support; BRIGHTER missing/nonbinary; control | 4 | 1 | 1 | 4 |
| Total | | 12 | 1 | 1 | 12 |

---

## 7. SECTION 6.8 — UPDATE EXTERNAL VALIDITY

FIND:
"Three tasks and both original/repaired variants share one
implementation and therefore do not constitute independent software
replications."

REPLACE:
"The two codebases examined here share no code, differ in authorship,
and target different tasks. The scheme identified full compliance in
three contracts and partial compliance in one for the second codebase;
this contrast supports transferability within the bounded case.
However, no second auditor, no independent project selection, and no
fault-detection accuracy estimate are provided. Generalization beyond
these two cases remains untested."

---

## 8. SECTION 6.6 — UPDATE CONTROLS INTRO

INSERT after "six release controls for teams developing...":

"The cross-codebase audit in Section 5.3 shows that a reference
implementation (HuggingFace transformers text-classification example)
satisfies three of these contracts by default, providing evidence
that the controls are achievable in practice."

---

## 9. RELATED WORK — ADD ACM ARTIFACT BADGING

INSERT after MLOps paragraph:

"ACM Artifact Badging distinguishes three review levels---Available,
Evaluated, and Reusable---for computational artifacts [REF]. Our
proposed release gates align with the Evaluated tier for the specific
contracts examined here, although we do not claim full artifact-reuse
criteria."

Reference:
[REF] ACM, Artifact Review and Badging - Version 1.1, 2020.
URL https://www.acm.org/publications/policies/artifact-review-and-badging-current

---

## 10. DATA AND CODE AVAILABILITY — UPDATE

FIND:
"Training checkpoints, predictions, and complete original execution
environments remain unavailable."

REPLACE:
"Training checkpoints, predictions, and complete original execution
environments remain unavailable. The scripts and oracle fixtures are
deposited at the repository above; the second-codebase audit artifacts
are in data/codebase_B_hf/."

---

## 11. TITLE — OPTIONAL (recommended)

CURRENT:
"Software Contracts and Experimental Validity in Machine-Learning
Pipelines: An Embedded Case Study of Centralized and Federated Training"

ALTERNATIVE:
"Software Contracts and Experimental Validity in Machine-Learning
Pipelines: A Two-Codebase Embedded Case Study"

Rationale: "Two-Codebase" adds credibility for external validity.

---

## CHECKLIST

- [ ] Fix 4 typos
- [ ] Update abstract ending
- [ ] Replace highlights
- [ ] Add Section 5.2 PyTorch subsection
- [ ] Insert Section 5.3
- [ ] Update Table 5
- [ ] Update Section 6.8
- [ ] Update Section 6.6
- [ ] Add ACM Artifact Badging ref
- [ ] Update Data and Code Availability
- [ ] Optional title change
- [ ] Recompile PDF
- [ ] Proofread
- [ ] Submit to IST
