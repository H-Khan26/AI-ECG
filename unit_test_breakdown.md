### test_io_contracts.py - Data ingestion & preprocessing contracts 
- Why: Lead data is commonly stored in both 8 and 12 lead formats. If lead order, sampling, or normalization drift, downstream training is corrupted. 
-  Assertion: parsers yield (8, L) float tensors, finite values, and lead_order matches a canonical spec. Normalization is checked statistically (per-lead ≈0 mean, ≈1 std with tolerant bounds). The 4 additional lead data can be derived from the first 8
- Benefit: early, cheap detection of schema changes; protects reproducibility and clinical validity. Minimizes size with 8 leads
--- 

### test_augmentations.py - Augmentation invariants 
- Why: Time shifts or noise must never alter shapes or label semantics. Multi-lead coherence matters physiologically (a shift should move all leads together when specified).
- Assertion: shape/dtype/finite invariants, determinism under seed, and synchronous vs per-lead behavior.
- Benefit: codifies augmentation “safety rails” so experiments remain comparable across branches
---

### test_model_contracts.py - Model contract
- Why: Model edits often break tensor plumbing or silently freeze gradients.
- Assertion: forward returns (B,C) logits from (B,8,L) inputs; backprop yields non-zero grads; parameter count stays within a sane ceiling; and a 25-step toy training loop produces a ≥30% loss drop.
- Benefit: guarantees a minimum level of learnability and protects against accidental architectural regressions—without touching private data.
---

### conftest.py - Formatting & readability:
Tests are short, named for intent, and skip gracefully if a feature isn’t present
A small set of helpers in conftest.py discovers model class and common function names, so the suite stays robust to renames.
All tests are fast (sub-second to a few seconds) and CI-ready on CPU
