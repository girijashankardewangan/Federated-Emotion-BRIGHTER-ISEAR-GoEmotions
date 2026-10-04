Dear Editors,

We submit the manuscript "Software Contracts and Experimental Validity in Machine-Learning Pipelines: A Two-Codebase Embedded Case Study" for consideration in Information and Software Technology.

SUMMARY
This work contributes to the journal's interest in empirical software engineering, software testing, and verification & validation. We conduct a retrospective embedded case study of 80 released machine-learning training runs and identify concrete contract violations in checkpoint handling (mutable state_dict aliasing), federated validation semantics (accepted-but-unused val_df), and task schema construction (silent zero-fill of missing labels).

FIT WITH IST SCOPE
The manuscript addresses multiple IST scope areas:
- Software testing and V&V: executable regression oracles for ML pipelines
- Empirical studies of SE: embedded case study with 12 bounded fixtures
- Software quality and metrics: contract checks SC1-SC7
- Software processes: proposed release gates for research software

KEY CONTRIBUTIONS
1. A finding-to-contract evidence register (SC1-SC7) mapping source-visible defects, recorded anomalies, and missing evidence to specific verification requirements.
2. Twelve bounded fixtures (nine NumPy-backed, three PyTorch-backed) demonstrating that the proposed oracles detect real defects: the original source fails eleven of twelve defect-focused fixtures while a candidate repair passes all twelve.
3. A cross-codebase application on HuggingFace's official text-classification example, showing that the four-layer scheme distinguishes compliant from defect-bearing implementations.

NOVELTY AND LIMITS
The manuscript does not claim a new contract formalism, a new learning algorithm, or a validated general audit framework. It contributes a case-derived, evidence-bounded connection between released score patterns, inspectable implementation defects, and actionable verification controls. Full PyTorch integration on training functions, corrected-training utility, and independent adjudication remain open; these are stated explicitly.

SUPPLEMENTARY MATERIAL
The submission package includes scripts, repair patches, fixture outputs, and the second-codebase audit artifacts. All scripts run without training or dataset downloads.

We believe this work is a strong fit for IST's empirical software engineering focus.

Sincerely,
Girija Shankar Dewangan (corresponding author)
Partha Roy
Rajesh Tiwari
