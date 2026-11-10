"""Feature engineering. Every feature is available at order_approved_at (prediction time)."""
import numpy as np
import pandas as pd

from . import config as C
from .numpy_ops import days_between, safe_ratio

NUMERIC = [
    "n_items", "n_products", "n_sellers", "n_categories", "total_price", "total_freight",
    "freight_ratio", "max_item_price", "total_weight_g", "total_volume_cm3",
    "planned_days", "approval_delay_hours", "approved_hour", "approved_dow", "approved_month",
    "same_state", "multi_seller", "total_payment", "n_payment_records", "max_installments",
    "n_payment_types", "seller_prior_orders",
]
CATEGORICAL = ["customer_state", "seller_state", "main_category", "main_payment_type"]
LOG_COLS = ["total_price", "total_freight", "max_item_price", "total_weight_g",
            "total_volume_cm3", "total_payment", "seller_prior_orders", "freight_ratio"]
FEATURES = NUMERIC + CATEGORICAL

# name -> (computation, source tables, rationale, availability)
FEATURE_DOC = {
    "n_items": ("count of order_items rows per order", "order_items", "bigger baskets need more handling/packing", "approval (items known)"),
    "n_products": ("distinct product_id per order", "order_items", "multi-product orders ship from more picks", "approval"),
    "n_sellers": ("distinct seller_id per order", "order_items", "multi-seller orders need consolidation", "approval"),
    "multi_seller": ("1 if n_sellers > 1", "order_items", "explicit flag for multi-seller orders (kept: primary-seller state is used for the others)", "approval"),
    "n_categories": ("distinct English category per order", "order_items, products, translation", "basket heterogeneity", "approval"),
    "total_price": ("sum(price)", "order_items", "order value / product type proxy", "approval"),
    "total_freight": ("sum(freight_value)", "order_items", "freight is distance/weight driven", "approval"),
    "freight_ratio": ("total_freight / total_price (np.where guarded)", "order_items", "expensive shipping relative to value suggests far/heavy items", "approval"),
    "max_item_price": ("max(price)", "order_items", "premium items may get special handling", "approval"),
    "total_weight_g": ("sum(product_weight_g); zero weights -> NaN -> train median", "products", "heavier parcels slower", "approval"),
    "total_volume_cm3": ("sum(L*H*W)", "products", "bulky parcels slower", "approval"),
    "planned_days": ("(estimated_delivery_date - order_approved_at) in days (np.timedelta64)", "orders", "length of the promised window; tight promises are easier to miss", "approval (estimate shown to buyer)"),
    "approval_delay_hours": ("order_approved_at - order_purchase_timestamp, hours", "orders", "payment/fraud-check friction", "approval"),
    "approved_hour": ("hour of order_approved_at", "orders", "late-day approvals miss same-day dispatch", "approval"),
    "approved_dow": ("weekday of order_approved_at (0=Mon)", "orders", "weekend approvals dispatch later", "approval"),
    "approved_month": ("month of order_approved_at", "orders", "seasonality (holidays, promotions)", "approval"),
    "same_state": ("1 if primary seller state == customer state", "sellers, customers", "intra-state routes are shorter", "approval"),
    "customer_state": ("customer_state", "customers", "destination region logistics", "approval"),
    "seller_state": ("state of the primary seller (most items; ties: highest value, then seller_id)", "sellers", "origin region logistics", "approval"),
    "main_category": ("dominant English category (most items; ties: higher value, then alphabetical)", "products, translation", "category-specific shipping", "approval"),
    "total_payment": ("sum(payment_value)", "order_payments", "includes freight+price (+ interest); payment known at approval", "approval (payment approved)"),
    "n_payment_records": ("count of payment rows", "order_payments", "split payments", "approval"),
    "max_installments": ("max(payment_installments)", "order_payments", "financing behaviour proxy", "approval"),
    "n_payment_types": ("distinct payment_type", "order_payments", "mixed payments (e.g. voucher + card)", "approval"),
    "main_payment_type": ("payment type with the largest value", "order_payments", "boleto vs card clears differently", "approval"),
    "seller_prior_orders": ("expanding count of earlier-approved orders of the primary seller (shift/cumcount; no outcomes used)", "orders, order_items", "experience/volume of the seller; uses ONLY earlier timestamps", "approval"),
}
EXCLUDED_DOC = [
    ("seller/customer historical late rate", "Needs outcomes of earlier orders that were not yet delivered at approval time -> high leakage risk; excluded."),
    ("shipping_limit_date", "Seller-side deadline; availability at prediction time not guaranteed; excluded from baseline model."),
    ("order_status, order_delivered_*_date, delivery_days, delay_days", "Post-approval outcomes (forbidden)."),
]


def build_features(m: pd.DataFrame) -> pd.DataFrame:
    """m = merged order-level table (ALL orders). Adds feature columns."""
    f = m.copy()
    f["freight_ratio"] = safe_ratio(f["total_freight"], f["total_price"])
    est = f["order_estimated_delivery_date"].dt.normalize()
    f["planned_days"] = days_between(est, f["order_approved_at"])
    f["approval_delay_hours"] = days_between(f["order_approved_at"], f["order_purchase_timestamp"]) * 24
    f["approved_hour"] = f["order_approved_at"].dt.hour
    f["approved_dow"] = f["order_approved_at"].dt.day_of_week
    f["approved_month"] = f["order_approved_at"].dt.month
    f["multi_seller"] = (f["n_sellers"] > 1).astype("float64").where(f["n_sellers"].notna())
    f["same_state"] = np.where(f["seller_state"].notna() & f["customer_state"].notna(),
                               (f["seller_state"] == f["customer_state"]).astype(float), np.nan)

    # Seller experience: number of orders of the same primary seller approved strictly earlier.
    tmp = f[["order_id", "primary_seller_id", "order_approved_at"]].dropna()
    tmp = tmp.sort_values(["primary_seller_id", "order_approved_at", "order_id"])
    tmp["seller_prior_orders"] = tmp.groupby("primary_seller_id").cumcount()
    # orders with the identical approval timestamp: count only strictly earlier ones
    grp = tmp.groupby(["primary_seller_id", "order_approved_at"])["seller_prior_orders"].transform("min")
    tmp["seller_prior_orders"] = grp
    f = f.merge(tmp[["order_id", "seller_prior_orders"]], on="order_id", how="left", validate="one_to_one")
    return f


def write_feature_dictionary(path) -> None:
    lines = ["# Feature dictionary", "",
             "Prediction time = `order_approved_at`. Items, payment and the estimated delivery date are "
             "assumed known at that moment (documented assumption).", "",
             "| Feature | How computed | Source table(s) | Why relevant | Available when |",
             "|---|---|---|---|---|"]
    for k in FEATURES:
        a, b, c, d = FEATURE_DOC[k]
        lines.append(f"| `{k}` | {a} | {b} | {c} | {d} |")
    lines += ["", "## Considered and excluded", "", "| Item | Reason |", "|---|---|"]
    lines += [f"| {a} | {b} |" for a, b in EXCLUDED_DOC]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
