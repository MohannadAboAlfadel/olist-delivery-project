"""Table audits (before / after cleaning), record-count check and FK integrity."""
import pandas as pd

from . import config as C

KEYS = {
    "orders": ["order_id"],
    "customers": ["customer_id"],
    "order_items": ["order_id", "order_item_id"],
    "order_payments": ["order_id", "payment_sequential"],
    "products": ["product_id"],
    "sellers": ["seller_id"],
    "translation": ["product_category_name"],
}


def audit_tables(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (table-level summary, column-level summary)."""
    t_rows, c_rows = [], []
    for name, df in tables.items():
        key = KEYS[name]
        t_rows.append(
            {
                "table": name,
                "rows": len(df),
                "cols": df.shape[1],
                "total_missing_cells": int(df.isna().sum().sum()),
                "full_duplicate_rows": int(df.duplicated().sum()),
                "key": "+".join(key),
                "key_unique": bool(not df.duplicated(subset=key).any()),
                "duplicate_keys": int(df.duplicated(subset=key).sum()),
            }
        )
        for col in df.columns:
            n_miss = int(df[col].isna().sum())
            c_rows.append(
                {
                    "table": name,
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "n_missing": n_miss,
                    "pct_missing": round(100 * n_miss / len(df), 3),
                    "n_unique": int(df[col].nunique(dropna=True)),
                }
            )
    return pd.DataFrame(t_rows), pd.DataFrame(c_rows)


def record_count_check(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [
        {
            "table": n,
            "expected": C.EXPECTED_COUNTS[n],
            "actual": len(tables[n]),
            "match": len(tables[n]) == C.EXPECTED_COUNTS[n],
        }
        for n in C.EXPECTED_COUNTS
    ]
    return pd.DataFrame(rows)


def fk_orphans(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Count child rows whose foreign key has no parent."""
    checks = [
        ("orders.customer_id -> customers", t["orders"]["customer_id"], t["customers"]["customer_id"]),
        ("order_items.order_id -> orders", t["order_items"]["order_id"], t["orders"]["order_id"]),
        ("order_payments.order_id -> orders", t["order_payments"]["order_id"], t["orders"]["order_id"]),
        ("order_items.product_id -> products", t["order_items"]["product_id"], t["products"]["product_id"]),
        ("order_items.seller_id -> sellers", t["order_items"]["seller_id"], t["sellers"]["seller_id"]),
        (
            "products.category -> translation",
            t["products"]["product_category_name"].dropna(),
            t["translation"]["product_category_name"],
        ),
    ]
    rows = []
    for name, child, parent in checks:
        orphans = int((~child.isin(set(parent))).sum())
        rows.append({"relation": name, "child_rows": len(child), "orphans": orphans})
    return pd.DataFrame(rows)
