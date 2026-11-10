# Peer review checklist

Reviewer follows the cyclic plan in `docs/CONTRIBUTIONS.md`. Tick each item and leave comments on the branch.

## 1. Data quality & cleaning (`feature/data-quality`)
- [ ] Raw input is never modified; outputs go to `data/processed/` only.
- [ ] Every drop/imputation is in `docs/decisions_log.md` with reason and count.
- [ ] Repeated `order_id` in items/payments is NOT treated as a duplicate.
- [ ] Zip prefixes are strings; dates parsed with `errors="coerce"` and failures counted.
- [ ] Record counts match expected values; FK orphan report is present.

## 2. Merge & features (`feature/merge-features`)
- [ ] Items and payments are aggregated to order level BEFORE merging; merges use `validate=`.
- [ ] Model table has exactly one row per eligible order (`order_id` unique).
- [ ] Multi-seller and multi-category rules are deterministic and documented.
- [ ] Every feature is available at `order_approved_at` (see `feature_dictionary.md`).
- [ ] Target uses calendar-day comparison; eligibility funnel is documented.

## 3. Analysis & figures (`feature/eda`)
- [ ] Each of the 6 questions has a table, a figure and a non-descriptive interpretation.
- [ ] Group comparisons show n; groups with n < 100 are not ranked or are flagged.
- [ ] Every figure states its population (denominator); axis labels and units are present.
- [ ] No causal wording ("causes"); delivery outcomes only used descriptively.

## 4. Models & evaluation (`feature/modeling`)
- [ ] Temporal split on `order_approved_at`, no shuffling; boundaries and counts reported.
- [ ] Imputer/encoder/scaler live inside Pipelines fitted on train only.
- [ ] Hyper-parameters and threshold chosen on validation only; test evaluated once per model.
- [ ] Baseline present; precision/recall/F1/AP/confusion matrix reported, not only accuracy.
- [ ] Experiments log complete (model, params, features, metrics, seed).

## 5. Tests & documentation (`feature/tests-docs`)
- [ ] `pytest -q` passes from a clean checkout after `python run_all.py`.
- [ ] Tests cover key uniqueness, no row duplication, sums, target cases, forbidden columns, split order.
- [ ] README commands work; versions match `requirements.txt`; source and licence are cited.
- [ ] Report numbers match `outputs/` (spot-check at least five numbers).
