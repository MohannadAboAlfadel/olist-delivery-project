"""Cleaning with a decision log. No row is dropped or imputed without a logged reason."""
import numpy as np
import pandas as pd

from . import config as C
from .numpy_ops import invalid_value_mask


class DecisionLog:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, area: str, decision: str, reason: str, n_affected: int) -> None:
        self.rows.append(
            {"area": area, "decision": decision, "reason": reason, "n_affected": int(n_affected)}
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def clean_tables(raw: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], DecisionLog]:
    log = DecisionLog()
    t = {k: v.copy() for k, v in raw.items()}

    # ---- full duplicate rows (NOT repeated order_id, which is legitimate 1:N) ----
    for name, df in t.items():
        n = int(df.duplicated().sum())
        if n:
            t[name] = df.drop_duplicates().reset_index(drop=True)
        log.add("duplicates", f"{name}: drop rows identical in all columns" if n else
                f"{name}: no fully identical rows - nothing dropped",
                "Repeated order_id in items/payments is legitimate (multi-item / multi-payment orders); "
                "only all-column duplicates are errors.", n)

    # ---- orders: dates ----
    o = t["orders"]
    for col in C.DATE_COLS:
        before_na = int(o[col].isna().sum())
        o[col] = pd.to_datetime(o[col], errors="coerce")
        failed = int(o[col].isna().sum()) - before_na
        log.add("dates", f"orders.{col} -> datetime (errors='coerce')",
                "Needed for duration/target arithmetic; unparseable values become NaT.", failed)
    for col, n in o[C.DATE_COLS].isna().sum().items():
        if n:
            log.add("missing", f"orders.{col}: {n} missing kept as NaT (no imputation)",
                    "Missing delivery/approval dates mean the event did not happen (or is unknown); "
                    "filling them would fabricate the outcome and leak the target.", n)

    # ---- orders: event-order checks (flag, do not drop) ----
    flags = {
        "flag_approved_before_purchase": o["order_approved_at"] < o["order_purchase_timestamp"],
        "flag_carrier_before_approved": o["order_delivered_carrier_date"] < o["order_approved_at"],
        "flag_carrier_before_purchase": o["order_delivered_carrier_date"] < o["order_purchase_timestamp"],
        "flag_delivered_before_carrier": o["order_delivered_customer_date"] < o["order_delivered_carrier_date"],
        "flag_delivered_before_purchase": o["order_delivered_customer_date"] < o["order_purchase_timestamp"],
    }
    for name, f in flags.items():
        o[name] = f.fillna(False).astype(bool)
        log.add("chronology", f"{name}: flagged, rows kept",
                "Out-of-order timestamps are recorded; carrier-date anomalies do not affect the "
                "target or features. Only purchase/approval/delivered-before-purchase anomalies "
                "remove an order from the *eligible* set (see target funnel).", int(o[name].sum()))
    t["orders"] = o

    # ---- zip prefixes: identifiers, left-padded strings ----
    for tbl, col in (("customers", "customer_zip_code_prefix"), ("sellers", "seller_zip_code_prefix")):
        t[tbl][col] = t[tbl][col].astype("string").str.zfill(5)
        log.add("types", f"{tbl}.{col}: kept as 5-char string (zfill)",
                "Input stored zip prefixes as integers, dropping leading zeros; identifiers must not be numeric.",
                len(t[tbl]))

    # ---- items: dates and impossible values ----
    it = t["order_items"]
    it["shipping_limit_date"] = pd.to_datetime(it["shipping_limit_date"], errors="coerce")
    log.add("dates", "order_items.shipping_limit_date -> datetime",
            "Parsed for completeness; NOT used as a model input (seller deadline, availability at "
            "approval not guaranteed).", int(it["shipping_limit_date"].isna().sum()))
    n_price = int(invalid_value_mask(it["price"]).sum())
    n_freight = int(invalid_value_mask(it["freight_value"], allow_zero=True).sum())
    n_freight0 = int((it["freight_value"] == 0).sum())
    log.add("values", "order_items.price <= 0: none found / none changed", "Checked with a NumPy boolean mask.", n_price)
    log.add("values", "order_items.freight_value < 0: none found / none changed", "Checked with a NumPy boolean mask.", n_freight)
    log.add("values", "order_items.freight_value == 0 kept", "Zero freight = free shipping, a valid business value.", n_freight0)
    t["order_items"] = it

    # ---- payments ----
    pay = t["order_payments"]
    n_neg = int(invalid_value_mask(pay["payment_value"], allow_zero=True).sum())
    n_zero = int((pay["payment_value"] == 0).sum())
    log.add("values", "order_payments.payment_value < 0: none", "NumPy mask check.", n_neg)
    log.add("values", "order_payments.payment_value == 0 kept", "Vouchers / not_defined payments; deleting would lose payment-method info.", n_zero)
    n_inst0 = int((pay["payment_installments"] < 1).sum())
    log.add("values", "payment_installments < 1 kept", "Records with 0 installments are rare data-entry quirks; kept as-is, tree/linear models are robust to them.", n_inst0)

    # ---- products ----
    p = t["products"]
    n_unknown = int((p["product_category_name"] == "unknown").sum())
    p["category_was_missing"] = p["product_category_name"].eq("unknown")
    log.add("missing", "products.product_category_name == 'unknown' kept as its own category (flagged)",
            "Input already replaced the 610 missing categories (and their descriptive fields, with global "
            "medians) - documented in docs/upstream/upstream_cleaning_decisions.md. We cannot recover the "
            "original NaNs of the descriptive fields; they are NOT used as model features.", n_unknown)
    for col in ("product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"):
        mask = invalid_value_mask(p[col])
        log.add("values", f"products.{col} <= 0 -> NaN" if mask.any() else f"products.{col} <= 0: none",
                "A zero weight/dimension is physically impossible; set to NaN so the model imputer "
                "(fitted on TRAIN only) fills it.", int(mask.sum()))
        p[col] = p[col].where(~mask, np.nan)
    p["product_volume_cm3"] = p["product_length_cm"] * p["product_height_cm"] * p["product_width_cm"]
    t["products"] = p

    # ---- translation fallback ----
    tr = t["translation"]
    missing_tr = ~p["product_category_name"].isin(set(tr["product_category_name"]))
    miss_names = sorted(p.loc[missing_tr, "product_category_name"].dropna().unique())
    extra = pd.DataFrame({"product_category_name": miss_names, "product_category_name_english": miss_names})
    t["translation"] = pd.concat([tr, extra], ignore_index=True)
    log.add("categories", f"translation: added identity mapping for {len(miss_names)} untranslated categories {miss_names}",
            "Categories absent from the 71-row translation table keep their Portuguese name instead of NaN.",
            int(missing_tr.sum()))
    return t, log


def write_decisions_log(log: DecisionLog, extra_sections: dict[str, str] | None = None) -> None:
    df = log.to_frame()
    lines = [
        "# Decisions log (auto-generated by `python run_all.py`)",
        "",
        "Every cleaning / analysis decision with reason and number of affected rows.",
        "The upstream (teammate) cleaning step that produced the input CSVs is documented in "
        "`docs/upstream/upstream_cleaning_decisions.md`.",
        "",
        "## A. Cleaning decisions",
        "",
        "| Area | Decision | Reason | Affected |",
        "|---|---|---|---:|",
    ]
    for r in df.itertuples():
        lines.append(f"| {r.area} | {r.decision} | {r.reason} | {r.n_affected:,} |")
    for title, body in (extra_sections or {}).items():
        lines += ["", f"## {title}", "", body]
    (C.DOCS / "decisions_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
