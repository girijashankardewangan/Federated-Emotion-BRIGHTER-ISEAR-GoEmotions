
### 5.3. Cross-codebase application of the four-layer scheme

To assess whether the four-layer scheme can distinguish defect-bearing 
from compliant implementations, we applied the same audit procedure to 
a second open-source codebase: HuggingFace's official text-classification 
example (`examples/pytorch/text-classification/run_glue.py`, 646 lines), 
which differs in authorship, framework version, and target task.

Table X summarises the audit outcomes.

**Table X. Cross-codebase contract comparison.**

| Contract | Codebase A (BRIGHTER) | Codebase B (HF example) |
|---|---|---|
| SC1 checkpoint immutability | Violated (line 209, aliased state_dict) | Compliant (Trainer handles selection) |
| SC2 validation semantics | Unused (val_df accepted, never read) | Compliant (eval_dataset passed and used) |
| SC3 schema validation | Silent zero-fill for missing labels | Compliant (explicit label-to-id check, lines 415-426) |
| SC4 provenance | Missing (no revision pinning) | Partial (model_revision supported, not enforced) |

**SC1 (checkpoint immutability).** The HF example delegates model 
selection and saving to the `Trainer` class (line 537), which internally 
preserves the selected state and serializes it via `save_model()` (line 
559). No direct `state_dict()` assignment appears in the audited file. 
The scheme therefore classifies this codebase as compliant.

**SC2 (validation semantics).** The example constructs `eval_dataset` 
from the `validation` split (line 485), passes it to the Trainer (line 
541), and uses it for evaluation (line 582). Unlike Codebase A, the 
validation argument is genuinely consumed for model selection.

**SC3 (schema validation).** The example derives `label_list` and 
`num_labels` directly from the dataset feature names (lines 343-344) 
and validates consistency against the model's `label2id` map (lines 
415-426), raising an explicit error on mismatch. No silent zero-fill 
occurs.

**SC4 (provenance).** The example exposes a `model_revision` argument 
(line 200-202) and propagates it to model, tokenizer, and config 
loading (lines 368, 376, 385). This is partial provenance support: the 
mechanism exists but is optional and not enforced by default.

**Interpretation.** The scheme distinguishes a defect-bearing release 
(Codebase A) from a well-engineered reference implementation (Codebase 
B) using the same four-layer procedure and observable source artifacts. 
This supports the scheme's diagnostic utility across codebases, while 
SC4's partial status shows that compliance is graded rather than binary. 
The bounded scope of this single cross-codebase test does not establish 
fault-detection accuracy; broader validation across many projects and 
independent auditors remains necessary.
