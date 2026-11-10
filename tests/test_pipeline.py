import numpy as np
import pandas as pd

from src import audit, config as C, features as F, modeling
from src.target import compute_is_late


# ---------------------------------------------------------------- keys & integrity
def test_primary_keys_unique(clean):
    assert clean["orders"]["order_id"].is_unique
    assert clean["customers"]["customer_id"].is_unique
    assert clean["products"]["product_id"].is_unique
    assert clean["sellers"]["seller_id"].is_unique
    assert not clean["order_items"].duplicated(["order_id", "order_item_id"]).any()
    assert not clean["order_payments"].duplicated(["order_id", "payment_sequential"]).any()


def test_record_counts_match_expected(raw):
    assert audit.record_count_check(raw)["match"].all()


def test_no_orphan_foreign_keys(clean):
    fk = audit.fk_orphans(clean)
    assert (fk[~fk.relation.str.contains("translation")]["orphans"] == 0).all()


def test_zip_prefix_is_5_char_string(clean):
    assert clean["customers"]["customer_zip_code_prefix"].str.len().eq(5).all()
    assert clean["sellers"]["seller_zip_code_prefix"].str.len().eq(5).all()


# ---------------------------------------------------------------- merge
def test_merge_does_not_duplicate_orders(merged, clean):
    assert merged["order_id"].is_unique
    assert len(merged) == len(clean["orders"])


def test_model_table_one_row_per_eligible_order(model_table, full):
    funnel = full.attrs["funnel"]
    assert model_table["order_id"].is_unique
    assert len(model_table) == int(funnel["orders_remaining"].iloc[-1])
    assert model_table["is_late"].notna().all()
    assert (model_table["order_status"] == "delivered").all()


def test_aggregated_sums_match_raw(merged, clean):
    m = merged.set_index("order_id")
    it = clean["order_items"].groupby("order_id")
    price, freight = it["price"].sum(), it["freight_value"].sum()
    idx = price.index
    assert np.allclose(m.loc[idx, "total_price"], price, atol=1e-6)
    assert np.allclose(m.loc[idx, "total_freight"], freight, atol=1e-6)
    assert np.allclose(m.loc[idx, "n_items"], it.size())
    pay = clean["order_payments"].groupby("order_id")["payment_value"].sum()
    assert np.allclose(m.loc[pay.index, "total_payment"], pay, atol=1e-6)
    # grand totals unchanged by the merge (no multiplication)
    assert np.isclose(m["total_price"].sum(), clean["order_items"]["price"].sum())
    assert np.isclose(m["total_payment"].sum(), clean["order_payments"]["payment_value"].sum())


# ---------------------------------------------------------------- target
def _ts(s):
    return pd.Series(pd.to_datetime(s))


def test_target_definition_cases():
    est = _ts(["2018-01-10"] * 6)
    dlv = _ts([
        "2018-01-10 00:00:00",   # on the estimated day -> 0
        "2018-01-11 00:00:01",   # day after -> 1
        None,                    # missing -> NaN
        "2018-01-05 12:00:00",   # before -> 0
        "2018-01-10 23:59:59",   # same day, hours later -> 0
        "2018-01-10 08:30:00",   # same day -> 0
    ])
    res = compute_is_late(dlv, est)
    assert res.iloc[0] == 0 and res.iloc[1] == 1 and np.isnan(res.iloc[2])
    assert res.iloc[3] == 0 and res.iloc[4] == 0 and res.iloc[5] == 0


def test_missing_estimated_date_gives_no_target():
    res = compute_is_late(_ts(["2018-01-10"]), pd.Series(pd.to_datetime([None])))
    assert np.isnan(res.iloc[0])


# ---------------------------------------------------------------- forbidden columns
def test_forbidden_columns_not_in_features(model_table):
    X = modeling._X(model_table)
    cols = set(X.columns)
    assert not (cols & C.FORBIDDEN_COLUMNS)
    for bad in ("order_status", "order_delivered_customer_date", "order_delivered_carrier_date"):
        assert bad not in cols
    # no derived column built from delivery outcomes
    assert not [c for c in cols if "deliver" in c and c != "planned_days"]
    assert "is_late" not in cols and "actual_days" not in cols


def test_feature_dictionary_covers_all_features():
    assert set(F.FEATURES) <= set(F.FEATURE_DOC)
    assert len(F.FEATURES) >= 8


# ---------------------------------------------------------------- splits
def test_temporal_split_no_leakage(model_table):
    tr, va, te = modeling.temporal_split(model_table)
    assert tr.order_approved_at.max() < va.order_approved_at.min()
    assert va.order_approved_at.max() < te.order_approved_at.min()
    assert len(tr) + len(va) + len(te) == len(model_table)
    assert set(tr.order_id).isdisjoint(va.order_id) and set(va.order_id).isdisjoint(te.order_id)


def test_seller_prior_orders_uses_only_earlier_orders(full):
    one = full[full.primary_seller_id == full.primary_seller_id.dropna().iloc[0]].sort_values("order_approved_at")
    # strictly earlier count: first order of a seller has 0 prior orders; values never exceed row position
    assert one["seller_prior_orders"].iloc[0] == 0
    assert (one["seller_prior_orders"].to_numpy() <= np.arange(len(one))).all()
