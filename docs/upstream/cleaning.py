import pandas as pd
import numpy as np
import os
import pathlib
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw"
PROCESSED_PATH = BASE_DIR / "data" / "processed"

FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv"
}

def load_data():
    """قراءة البيانات"""
    tables = {}
    for name, filename in FILES.items():
        filepath = DATA_PATH / filename
        if filepath.exists():
            tables[name] = pd.read_csv(filepath)
        else:
            print(f"❌ لم يتم العثور على {filename}")
    return tables

def clean_data(tables):
    """تنظيف البيانات بناءً على القرارات الموثقة"""
    print("\n" + "="*50)
    print("🧹 بدء عملية التنظيف")
    print("="*50)
    
    cleaned = tables.copy()
    
    orders = cleaned["orders"]
    date_cols = [
        "order_purchase_timestamp", "order_approved_at", 
        "order_delivered_carrier_date", "order_delivered_customer_date", 
        "order_estimated_delivery_date"
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors='coerce')
        
   
    invalid_chrono = orders[orders['order_approved_at'] < orders['order_purchase_timestamp']]
    print(f"⚠️ عدد الطلبات ذات الترتيب الزمني الخاطئ (اعتماد قبل شراء): {len(invalid_chrono)}")
   
    
    for df_name in ["customers", "sellers"]:
        df = cleaned[df_name]
        zip_col = 'customer_zip_code_prefix' if df_name == 'customers' else 'seller_zip_code_prefix'
        df[zip_col] = df[zip_col].astype(str).str.zfill(5)
        
   
    products = cleaned["products"]
   
    products['product_category_name'] = products['product_category_name'].fillna('unknown')
   
    for col in ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']:
        median_val = products[col].median()
        products[col] = products[col].fillna(median_val)

  
    for col in ['product_name_lenght', 'product_description_lenght', 'product_photos_qty']:
        median_val = products[col].median()
        products[col] = products[col].fillna(median_val)
        
    
    items = cleaned["order_items"]
    items['shipping_limit_date'] = pd.to_datetime(items['shipping_limit_date'], errors='coerce')
   
    invalid_prices = items[items['price'] <= 0]
    print(f"⚠️ عدد العناصر ذات سعر صفر أو سالب: {len(invalid_prices)}")
    
    return cleaned

def run_tests(tables):
    """اختبارات آلية لضمان جودة البيانات (تفرد المفاتيح الأساسية)"""
    print("\n" + "="*50)
    print("🧪 تشغيل الاختبارات الآلية (Automated Tests)")
    print("="*50)
    
   
    assert tables['orders']['order_id'].is_unique, "❌ فشل: order_id مكرر!"
    print("✅ نجح: order_id فريد في جدول الطلبات.")
    
   
    assert tables['customers']['customer_id'].is_unique, "❌ فشل: customer_id مكرر!"
    print("✅ نجح: customer_id فريد في جدول العملاء.")
    
    
    assert tables['products']['product_id'].is_unique, "❌ فشل: product_id مكرر!"
    print("✅ نجح: product_id فريد في جدول المنتجات.")
    
    
    assert tables['sellers']['seller_id'].is_unique, "❌ فشل: seller_id مكرر!"
    print("✅ نجح: seller_id فريد في جدول البائعين.")
    
    
    assert tables['customers']['customer_zip_code_prefix'].str.len().max() == 5, "❌ فشل: الرموز البريدية ليست 5 خانات!"
    print("✅ نجح: الرموز البريدية تم تحويلها لـ String بطول 5 خانات.")
    
    print("🎉 جميع الاختبارات الآلية نجحت!")

def save_data(tables):
    """حفظ البيانات النظيفة"""
    print("\n" + "="*50)
    print("💾 حفظ البيانات النظيفة في data/processed/")
    print("="*50)
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    
    for name, df in tables.items():
        save_path = PROCESSED_PATH / f"{name}_cleaned.csv"
        df.to_csv(save_path, index=False)
        print(f"✅ تم حفظ: {name}_cleaned.csv")

if __name__ == "__main__":
    raw_tables = load_data()
    if raw_tables:
        cleaned_tables = clean_data(raw_tables)
        run_tests(cleaned_tables)
        save_data(cleaned_tables)