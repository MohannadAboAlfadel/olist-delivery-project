"""Reading the input CSVs into DataFrames with explicit dtypes."""
import pandas as pd

from . import config as C

_DTYPES = {
    "orders": {"order_id": "string", "customer_id": "string", "order_status": "string"},
    "customers": {
        "customer_id": "string",
        "customer_unique_id": "string",
        "customer_zip_code_prefix": "string",  # zip prefixes are identifiers, never numbers
        "customer_city": "string",
        "customer_state": "string",
    },
    "order_items": {"order_id": "string", "product_id": "string", "seller_id": "string"},
    "order_payments": {"order_id": "string", "payment_type": "string"},
    "products": {"product_id": "string", "product_category_name": "string"},
    "sellers": {
        "seller_id": "string",
        "seller_zip_code_prefix": "string",
        "seller_city": "string",
        "seller_state": "string",
    },
    "translation": {"product_category_name": "string", "product_category_name_english": "string"},
}


def md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table (no extra dependency)."""
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:,.4g}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v) for v in row) + " |")
    return "\n".join(out)


def load_raw_tables() -> dict[str, pd.DataFrame]:
    """Read the seven input tables. Date columns stay as text here (parsed in cleaning)."""
    tables = {}
    for name, fname in C.RAW_FILES.items():
        tables[name] = pd.read_csv(C.DATA_RAW / fname, dtype=_DTYPES[name])
    return tables
