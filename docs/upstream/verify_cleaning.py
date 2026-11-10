import pandas as pd
import pathlib
import os
import numpy as np

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
PROCESSED_PATH = BASE_DIR / "data" / "processed"


EXPECTED_ROWS = {
    "orders": 99441, "customers": 99441, "order_items": 112650,
    "payments": 103886, "products": 32951, "sellers": 3095,
    "category_translation": 71
}

def load(name):
    df = pd.read_csv(PROCESSED_PATH / f"{name}_cleaned.csv")
    return df

def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

def ok(msg): print(f"  ✅ {msg}")
def warn(msg): print(f"  ⚠️  {msg}")
def err(msg): print(f"  ❌ {msg}")


def verify_orders():
    section("📌 ORDERS")
    df = load("orders")
    print(f"  Row count: {len(df):,} (expected: {EXPECTED_ROWS['orders']:,})")
    if len(df) != EXPECTED_ROWS["orders"]: err("عدد الصفوف لا يطابق!")
    else: ok("عدد الصفوف صحيح.")


    date_cols = ['order_purchase_timestamp','order_approved_at',
                 'order_delivered_carrier_date','order_delivered_customer_date',
                 'order_estimated_delivery_date']
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors='coerce')
        n_nat = df[c].isna().sum()
        if n_nat == 0: ok(f"{c}: كل القيم صالحة كتواريخ.")
        else: warn(f"{c}: {n_nat} قيمة NaN (بما فيها المفقودة)")

  
    bad1 = (df['order_approved_at'] < df['order_purchase_timestamp']).sum()
    bad2 = (df['order_delivered_carrier_date'] < df['order_purchase_timestamp']).sum()
    bad3 = (df['order_delivered_customer_date'] < df['order_delivered_carrier_date']).sum()
    for label, val in [("approved < purchase", bad1),
                       ("carrier < purchase", bad2),
                       ("customer < carrier", bad3)]:
        if val == 0: ok(f"الترتيب الزمني ({label}): سليم.")
        else: warn(f"الترتيب الزمني ({label}): {val} سجل مخالف.")


    print(f"  قيم order_status: {sorted(df['order_status'].dropna().unique())}")

  
    if df['order_id'].is_unique: ok("order_id فريد.")
    else: err("order_id مكرر!")


def verify_customers():
    section("📌 CUSTOMERS")
    df = pd.read_csv(PROCESSED_PATH/"customers_cleaned.csv",
                     dtype={'customer_zip_code_prefix': str})
    if len(df) == EXPECTED_ROWS["customers"]: ok("عدد الصفوف صحيح.")
    else: err(f"عدد الصفوف: {len(df)}")

    lengths = df['customer_zip_code_prefix'].str.len().unique()
    if list(lengths) == [5]: ok("جميع الرموز البريدية بطول 5.")
    else: err(f"أطوال غير متوقعة: {sorted(lengths)}")

    invalid_state = df[~df['customer_state'].str.match(r'^[A-Z]{2}$', na=False)]
    if len(invalid_state) == 0: ok("customer_state: كلها حرفان كبيران.")
    else: warn(f"حالات شاذة في customer_state: {len(invalid_state)}")

    if df['customer_id'].is_unique: ok("customer_id فريد.")
    else: err("customer_id مكرر!")

    print(f"  عدد العملاء الفريدين (unique_id): {df['customer_unique_id'].nunique():,}")


def verify_order_items():
    section("📌 ORDER_ITEMS")
    df = load("order_items")
    if len(df) == EXPECTED_ROWS["order_items"]: ok("عدد الصفوف صحيح.")
    else: err(f"عدد الصفوف: {len(df)}")


    neg = (df['price'] < 0).sum()
    zero = (df['price'] == 0).sum()
    print(f"  price: <0 = {neg}, ==0 = {zero}")
    if neg == 0 and zero == 0: ok("الأسعار كلها موجبة.")
    else: warn("توجد قيم غير منطقية في price.")


    neg_f = (df['freight_value'] < 0).sum()
    if neg_f == 0: ok("freight_value غير سالب.")
    else: warn(f"freight_value سالب: {neg_f}")


    dup = df.duplicated(subset=['order_id','order_item_id']).sum()
    if dup == 0: ok("(order_id, order_item_id) فريد.")
    else: err(f"مكرر: {dup}")


    sl = pd.to_datetime(df['shipping_limit_date'], errors='coerce')
    ok(f"shipping_limit_date: {sl.isna().sum()} NaN.") if sl.isna().sum()==0 else warn("تواريخ شحن مفقودة")


def verify_payments():
    section("📌 PAYMENTS")
    df = load("payments")
    if len(df) == EXPECTED_ROWS["payments"]: ok("عدد الصفوف صحيح.")
    else: err(f"عدد الصفوف: {len(df)}")

    neg = (df['payment_value'] < 0).sum()
    zero = (df['payment_value'] == 0).sum()
    print(f"  payment_value: <0 = {neg}, ==0 = {zero}")
    if neg == 0 and zero == 0: ok("payment_value موجب.")
    else: warn("قيم دفع غير منطقية.")

    bad_inst = (df['payment_installments'] < 0).sum()
    if bad_inst == 0: ok("payment_installments غير سالب.")
    else: warn(f"أقساط سالبة: {bad_inst}")

    print(f"  قيم payment_type: {sorted(df['payment_type'].unique())}")

    dup = df.duplicated(subset=['order_id','payment_sequential']).sum()
    if dup == 0: ok("(order_id, payment_sequential) فريد.")
    else: err(f"مكرر: {dup}")


def verify_products():
    section("📌 PRODUCTS")
    df = load("products")
    if len(df) == EXPECTED_ROWS["products"]: ok("عدد الصفوف صحيح.")
    else: err(f"عدد الصفوف: {len(df)}")

  
    unk = (df['product_category_name'] == 'unknown').sum()
    print(f"  فئات 'unknown': {unk} (المتوقع 610)")
    if unk == 610: ok("الفئات المفقودة عُوضت بشكل صحيح.")
    else: warn("عدد 'unknown' لا يطابق 610.")

  
    total_missing = df.isna().sum().sum()
    if total_missing == 0: ok("لا توجد قيم مفقودة.")
    else: err(f"يوجد {total_missing} قيمة مفقودة.")

   
    print("\n  فحص القيم غير المنطقية (== 0):")
    for col in ['product_weight_g','product_length_cm',
                'product_height_cm','product_width_cm']:
        z = (df[col] == 0).sum()
        n = (df[col] < 0).sum()
        status = "✅" if z == 0 and n == 0 else "⚠️"
        print(f"    {status} {col}: ==0 {z}, <0 {n}")

  
    if df['product_id'].is_unique: ok("product_id فريد.")
    else: err("product_id مكرر!")

def verify_sellers():
    section("📌 SELLERS")
    df = pd.read_csv(PROCESSED_PATH/"sellers_cleaned.csv",
                     dtype={'seller_zip_code_prefix': str})
    if len(df) == EXPECTED_ROWS["sellers"]: ok("عدد الصفوف صحيح.")
    else: err(f"عدد الصفوف: {len(df)}")

    lengths = df['seller_zip_code_prefix'].str.len().unique()
    if list(lengths) == [5]: ok("جميع الرموز البريدية بطول 5.")
    else: err(f"أطوال: {sorted(lengths)}")

    if df['seller_id'].is_unique: ok("seller_id فريد.")
    else: err("seller_id مكرر!")


def verify_categories():
    section("📌 CATEGORY_TRANSLATION")
    df = load("category_translation")
    if len(df) == EXPECTED_ROWS["category_translation"]: ok("عدد الصفوف صحيح.")
    else: err(f"عدد الصفوف: {len(df)}")

    if df['product_category_name'].is_unique: ok("product_category_name فريد (لا تكرار).")
    else: err("product_category_name مكرر — يسبب مشاكل في الدمج!")

    if df.isna().sum().sum() == 0: ok("لا توجد قيم مفقودة.")
    else: warn("قيم مفقودة موجودة.")


if __name__ == "__main__":
    verify_orders()
    verify_customers()
    verify_order_items()
    verify_payments()
    verify_products()
    verify_sellers()
    verify_categories()

    print("\n" + "="*60)
    print("🏁 اكتمل التحقق الشامل")
    print("="*60)
    