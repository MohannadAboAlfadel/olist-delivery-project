"""Aggregate items and payments to ORDER level, then merge onto orders (1 row / order)."""
import numpy as np
import pandas as pd


def aggregate_items(items, products, sellers, translation) -> tuple[pd.DataFrame, dict]:
    p = products.merge(translation, on="product_category_name", how="left", validate="many_to_one")
    df = items.merge(p, on="product_id", how="left", validate="many_to_one")
    df = df.merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left", validate="many_to_one")

    g = df.groupby("order_id")
    agg = g.agg(
        n_items=("order_item_id", "size"),
        n_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        n_seller_states=("seller_state", "nunique"),
        n_categories=("product_category_name_english", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        max_item_price=("price", "max"),
        total_weight_g=("product_weight_g", "sum"),
        total_volume_cm3=("product_volume_cm3", "sum"),
    )

    # Dominant category: most items; ties -> highest price sum; then alphabetical.
    cat = (df.groupby(["order_id", "product_category_name_english"])
             .agg(k=("order_item_id", "size"), v=("price", "sum")).reset_index()
             .sort_values(["order_id", "k", "v", "product_category_name_english"],
                          ascending=[True, False, False, True]))
    main_cat = cat.drop_duplicates("order_id").set_index("order_id")["product_category_name_english"]

    # Primary seller: most items; ties -> highest price sum; then seller_id.
    sel = (df.groupby(["order_id", "seller_id", "seller_state"], dropna=False)
             .agg(k=("order_item_id", "size"), v=("price", "sum")).reset_index()
             .sort_values(["order_id", "k", "v", "seller_id"], ascending=[True, False, False, True]))
    primary = sel.drop_duplicates("order_id").set_index("order_id")[["seller_id", "seller_state"]]
    primary.columns = ["primary_seller_id", "seller_state"]

    out = agg.join(main_cat.rename("main_category")).join(primary).reset_index()
    stats = {
        "orders_with_items": len(out),
        "multi_seller_orders": int((out["n_sellers"] > 1).sum()),
        "multi_category_orders": int((out["n_categories"] > 1).sum()),
        "multi_seller_state_orders": int((out["n_seller_states"] > 1).sum()),
    }
    return out, stats


def aggregate_payments(pay) -> pd.DataFrame:
    g = pay.groupby("order_id")
    agg = g.agg(
        total_payment=("payment_value", "sum"),
        n_payment_records=("payment_sequential", "size"),
        max_installments=("payment_installments", "max"),
        n_payment_types=("payment_type", "nunique"),
    )
    t = (pay.groupby(["order_id", "payment_type"]).agg(v=("payment_value", "sum")).reset_index()
            .sort_values(["order_id", "v", "payment_type"], ascending=[True, False, True])
            .drop_duplicates("order_id").set_index("order_id")["payment_type"])
    return agg.join(t.rename("main_payment_type")).reset_index()


def build_order_table(tables) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (merged order-level table, merge-impact log, multi-seller stats)."""
    orders = tables["orders"]
    impact = [{"step": "orders (base)", "rows_after": len(orders), "unique_order_id": orders.order_id.nunique()}]

    m = orders.merge(tables["customers"], on="customer_id", how="left", validate="many_to_one")
    impact.append({"step": "+ customers (N:1 on customer_id)", "rows_after": len(m), "unique_order_id": m.order_id.nunique()})

    item_agg, stats = aggregate_items(tables["order_items"], tables["products"], tables["sellers"], tables["translation"])
    m = m.merge(item_agg, on="order_id", how="left", validate="one_to_one")
    impact.append({"step": "+ aggregated order_items (1:1 after groupby)", "rows_after": len(m), "unique_order_id": m.order_id.nunique()})

    pay_agg = aggregate_payments(tables["order_payments"])
    m = m.merge(pay_agg, on="order_id", how="left", validate="one_to_one")
    impact.append({"step": "+ aggregated order_payments (1:1 after groupby)", "rows_after": len(m), "unique_order_id": m.order_id.nunique()})

    assert m["order_id"].is_unique and len(m) == len(orders)
    stats["orders_without_items"] = int(m["n_items"].isna().sum())
    stats["orders_without_payment"] = int(m["total_payment"].isna().sum())
    return m, pd.DataFrame(impact), stats
