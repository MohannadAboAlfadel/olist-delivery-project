# Schema (ERD)

```mermaid
erDiagram
    CUSTOMERS ||--o{ ORDERS : "customer_id (1 record per order)"
    ORDERS ||--|{ ORDER_ITEMS : "order_id (1:N)"
    ORDERS ||--|{ ORDER_PAYMENTS : "order_id (1:N)"
    PRODUCTS ||--o{ ORDER_ITEMS : "product_id"
    SELLERS ||--o{ ORDER_ITEMS : "seller_id"
    TRANSLATION ||--o{ PRODUCTS : "product_category_name"
    CUSTOMERS { string customer_id PK string customer_unique_id string customer_zip_code_prefix string customer_state }
    ORDERS { string order_id PK string customer_id FK string order_status datetime order_purchase_timestamp datetime order_approved_at datetime order_delivered_customer_date datetime order_estimated_delivery_date }
    ORDER_ITEMS { string order_id PK_FK int order_item_id PK string product_id FK string seller_id FK float price float freight_value }
    ORDER_PAYMENTS { string order_id PK_FK int payment_sequential PK string payment_type int payment_installments float payment_value }
    PRODUCTS { string product_id PK string product_category_name FK float product_weight_g }
    SELLERS { string seller_id PK string seller_zip_code_prefix string seller_state }
    TRANSLATION { string product_category_name PK string product_category_name_english }
```

- `customer_id` is a per-order customer record; `customer_unique_id` identifies the real (possibly repeat) customer.
- An **order** is one purchase; an **order item** is one product line within it; a **payment record** is one payment instalment/method used for it.
