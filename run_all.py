"""Single entry point: runs every stage in order. Usage: python run_all.py"""
import json
import sys
import time

import pandas as pd

from src import audit, cleaning, config as C, eda, features, io_utils, merge, modeling, target


def main() -> None:
    t0 = time.time()
    C.ensure_dirs()

    # ---- (a) load + audit BEFORE ----
    raw = io_utils.load_raw_tables()
    counts = audit.record_count_check(raw)
    counts.to_csv(C.AUDIT_DIR / "record_counts_check.csv", index=False)
    assert counts["match"].all(), f"Unexpected record counts:\n{counts}"
    tb, cb = audit.audit_tables(raw)
    tb.to_csv(C.AUDIT_DIR / "audit_before_tables.csv", index=False)
    cb.to_csv(C.AUDIT_DIR / "audit_before_columns.csv", index=False)
    audit.fk_orphans(raw).to_csv(C.AUDIT_DIR / "fk_orphans_before.csv", index=False)
    print("[a] audit before done")

    # ---- (b) cleaning ----
    clean, log = cleaning.clean_tables(raw)
    ta, ca = audit.audit_tables(clean)
    ta.to_csv(C.AUDIT_DIR / "audit_after_tables.csv", index=False)
    ca.to_csv(C.AUDIT_DIR / "audit_after_columns.csv", index=False)
    fk_after = audit.fk_orphans(clean)
    fk_after.to_csv(C.AUDIT_DIR / "fk_orphans_after.csv", index=False)
    for name, df in clean.items():
        df.to_csv(C.DATA_PROCESSED / f"{name}_clean.csv", index=False)
    print("[b] cleaning done")

    # ---- (c) merge ----
    merged, impact, mstats = merge.build_order_table(clean)
    impact.to_csv(C.TABLE_DIR / "merge_impact.csv", index=False)
    pd.Series(mstats).to_csv(C.TABLE_DIR / "multi_seller_stats.csv", header=["value"])
    print("[c] merge done", mstats)

    # ---- (d,e) features + target ----
    full = features.build_features(merged)
    full = target.add_target(full)
    funnel, elig_mask = target.eligibility_funnel(full)
    funnel.to_csv(C.TABLE_DIR / "eligibility_funnel.csv", index=False)
    full["eligible"] = elig_mask
    full.to_csv(C.DATA_PROCESSED / "orders_all_features.csv", index=False)
    model_table = full[elig_mask].reset_index(drop=True)
    assert model_table["order_id"].is_unique, "model table must have one row per order"
    assert len(model_table) == int(funnel["orders_remaining"].iloc[-1])
    assert model_table["is_late"].notna().all()
    model_table.to_csv(C.DATA_PROCESSED / "model_table.csv", index=False)
    features.write_feature_dictionary(C.DOCS / "feature_dictionary.md")
    print(f"[d/e] model table rows = {len(model_table):,}; late rate = {model_table.is_late.mean():.4f}")

    # ---- (f) EDA ----
    eda_out = eda.run_eda(full, model_table, funnel)
    print("[f] EDA done")

    # ---- (g) modelling ----
    results = modeling.run_modeling(model_table)
    print("[g] modelling done")

    # ---- decisions log (auto) ----
    extra = {
        "B. Merge & multi-seller handling": (
            f"- Items and payments are aggregated to order level BEFORE merging; each merge is validated "
            f"(`outputs/tables/merge_impact.csv`).\n"
            f"- Multi-seller orders: {mstats['multi_seller_orders']:,}; multi-category: {mstats['multi_category_orders']:,}; "
            f"orders spanning >1 seller state: {mstats['multi_seller_state_orders']:,}.\n"
            f"- Decision: seller_state = state of the *primary seller* (most items; ties: higher value, then seller_id) "
            f"plus the flag `multi_seller`. Reason: a single deterministic value per order, reproducible, with the "
            f"multi-seller effect kept visible.\n"
            f"- Dominant category: most items, ties -> higher value -> alphabetical.\n"
            f"- Orders without items: {mstats['orders_without_items']:,}; without payment: {mstats['orders_without_payment']:,}."),
        "C. Target & eligibility": io_utils.md_table(funnel),
        "D. Temporal split & modelling decisions": results["decisions_md"],
    }
    cleaning.write_decisions_log(log, extra)
    (C.OUT / "results.json").write_text(json.dumps(results["summary"], indent=2, default=str), encoding="utf-8")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
