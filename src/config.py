"""Central configuration: paths, seeds, expected record counts, thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"
AUDIT_DIR = OUT / "audit"
DOCS = ROOT / "docs"

SEED = 42

# Raw (as delivered) file names -> logical table names.
RAW_FILES = {
    "orders": "orders_cleaned.csv",
    "customers": "customers_cleaned.csv",
    "order_items": "order_items_cleaned.csv",
    "order_payments": "payments_cleaned.csv",
    "products": "products_cleaned.csv",
    "sellers": "sellers_cleaned.csv",
    "translation": "category_translation_cleaned.csv",
}

EXPECTED_COUNTS = {
    "orders": 99_441,
    "customers": 99_441,
    "order_items": 112_650,
    "order_payments": 103_886,
    "products": 32_951,
    "sellers": 3_095,
    "translation": 71,
}

DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

# Temporal split proportions over eligible orders ordered by order_approved_at.
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # remaining 15% = final test (most recent)

MIN_GROUP_N = 100  # minimum group size to rank/display a late-rate in plots

# Columns that must never reach the model.
FORBIDDEN_COLUMNS = {
    "order_status",
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "is_late",
    "delivery_days",
    "delay_days",
    "shipping_limit_date",
}


def ensure_dirs() -> None:
    for d in (DATA_PROCESSED, FIG_DIR, TABLE_DIR, AUDIT_DIR, DOCS):
        d.mkdir(parents=True, exist_ok=True)
