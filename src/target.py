"""Target definition (is_late) and eligibility funnel."""
import numpy as np
import pandas as pd


def compute_is_late(delivered: pd.Series, estimated: pd.Series) -> pd.Series:
    """1 if the delivery DAY is after the estimated DAY, else 0; NaN if either date is missing.

    Dates are compared as calendar days (normalize), never as times vs midnight.
    """
    d, e = delivered.dt.normalize(), estimated.dt.normalize()
    res = (d > e).astype("float64")
    res[d.isna() | e.isna()] = np.nan
    return res


def eligibility_funnel(m: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sequential exclusion funnel. Returns (funnel table, boolean eligible mask)."""
    steps = [
        ("All orders", pd.Series(True, index=m.index), ""),
        ("order_status == delivered", m["order_status"].eq("delivered"),
         "Canceled / unavailable / in-transit orders have no final outcome; they are NOT class 0."),
        ("delivered date and estimated date present",
         m["order_delivered_customer_date"].notna() & m["order_estimated_delivery_date"].notna(),
         "Outcome cannot be computed without both dates."),
        ("purchase and approval timestamps present and approval >= purchase",
         m["order_purchase_timestamp"].notna() & m["order_approved_at"].notna()
         & ~m["flag_approved_before_purchase"],
         "Prediction time is the approval timestamp; it must be valid."),
        ("delivered on/after purchase", ~m["flag_delivered_before_purchase"],
         "Delivery before purchase is impossible."),
        ("order has >= 1 item", m["n_items"].notna(),
         "Item-level features are undefined without items."),
    ]
    rows, mask = [], pd.Series(True, index=m.index)
    prev = len(m)
    for name, cond, why in steps:
        mask &= cond
        n = int(mask.sum())
        rows.append({"step": name, "orders_remaining": n, "excluded_at_step": prev - n, "reason": why})
        prev = n
    return pd.DataFrame(rows), mask


def add_target(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    m["is_late"] = compute_is_late(m["order_delivered_customer_date"], m["order_estimated_delivery_date"])
    return m
