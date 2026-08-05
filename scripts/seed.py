"""Seed the business schema (PLAN.md §3.2, Phase 1).

Deterministic and idempotent. Every value derives from `generate_series` via
modular arithmetic rather than `random()`, so **1,840 active customers is a
fact, not a probability** — it's the answer to beat 1 of the demo, and it has to
be the same number every rehearsal.

Truncates before inserting, so running it twice is the same as running it once.
"""

import asyncio
import time

import psycopg
from psycopg.rows import dict_row

from app.settings import settings

CUSTOMERS = 2_000
PRODUCTS = 200
ORDERS = 6_000

# Every table this script owns. Truncated together so the FKs don't matter.
SEEDED = [
    "order_item", "orders", "product", "customer",
    "app_user", "role", "permission", "role_permission", "user_role",
    "api_key", "session", "audit_log", "feature_flag", "notification",
    "email_template", "email_queue", "webhook_delivery", "address",
    "country", "currency", "exchange_rate", "tax_rate", "shipping_method",
    "shipment", "warehouse", "inventory_level", "stock_movement", "supplier",
    "purchase_order", "purchase_order_item", "category", "product_category",
    "price_history", "discount_code", "cart", "cart_item", "payment",
    "refund", "invoice", "support_ticket",
]

# --------------------------------------------------------------------------
# The four that matter
# --------------------------------------------------------------------------

CUSTOMER_SQL = f"""
INSERT INTO customer (name, region, signed_up, deleted_at)
SELECT
  'Customer ' || i,

  -- TRAP 2. Region casing is inconsistent, and west is the worst of it:
  -- 350 'west', 100 'West', 50 'WEST'. `WHERE region = 'west'` silently
  -- drops 30% of the region.
  CASE i % 4
    WHEN 0 THEN CASE (i / 4) % 10 WHEN 9 THEN 'WEST'
                                  WHEN 8 THEN 'West'
                                  WHEN 7 THEN 'West'
                                  ELSE 'west' END
    WHEN 1 THEN CASE (i / 4) % 10 WHEN 9 THEN 'EAST' ELSE 'east' END
    WHEN 2 THEN 'north'
    ELSE        CASE (i / 4) % 5 WHEN 4 THEN 'South' ELSE 'south' END
  END,

  DATE '2022-01-01' + ((i * 7) % 1400),

  -- TRAP 1. Soft deletes on exactly 160 of {CUSTOMERS} rows, so the honest
  -- answer to "how many customers do we have?" is 1,840 and the obvious
  -- SELECT count(*) is wrong by 160.
  CASE WHEN i % 25 IN (0, 1)
       THEN TIMESTAMPTZ '2024-06-01' + ((i % 400) || ' days')::interval
  END
FROM generate_series(1, {CUSTOMERS}) AS i
"""

PRODUCT_SQL = f"""
INSERT INTO product (name, unit_price, discontinued)
SELECT
  'Product ' || i,
  -- TRAP 4, half of it: this is the price *now*. It is not what anything
  -- was actually sold for.
  ROUND((5 + (i % 180) * 1.37)::numeric, 2),
  i % 11 = 0
FROM generate_series(1, {PRODUCTS}) AS i
"""

ORDERS_SQL = f"""
INSERT INTO orders (customer_id, created, status)
SELECT
  ((i * 7) % {CUSTOMERS}) + 1,

  -- TRAP 5. The column is `created`. Every model reaches for `created_at`
  -- first, and `created_at` does not exist on this table.
  -- Anchored to today so "last quarter" always has data to find.
  date_trunc('day', now()) - ((i % 730) || ' days')::interval
                           + ((i % 24) || ' hours')::interval,

  -- TRAP 3. 12.5% cancelled. Revenue that includes them is overstated.
  CASE
    WHEN i % 8 = 0 THEN 'cancelled'
    WHEN i % 8 = 1 THEN 'refunded'
    WHEN i % 3 = 0 THEN 'shipped'
    WHEN i % 5 = 0 THEN 'pending'
    ELSE 'completed'
  END
FROM generate_series(1, {ORDERS}) AS i
"""

ORDER_ITEM_SQL = f"""
INSERT INTO order_item (order_id, product_id, qty, price)
SELECT
  o.id,
  ((o.id * 3 + k) % {PRODUCTS}) + 1,
  ((o.id + k) % 4) + 1,

  -- TRAP 4, the other half. What a line sold for is not today's list price.
  --
  -- Two sources of divergence, and the second one exists because of a bug:
  -- an earlier version only discounted orders over a year old, which meant
  -- the trap was *invisible* to beat 5 ("revenue by region last quarter") —
  -- the demo's flagship revenue question could join to product.unit_price and
  -- get exactly the right answer by luck. Promotions hit recent orders too.
  ROUND(
    p.unit_price
    * CASE WHEN o.created < now() - interval '1 year' THEN 0.80 ELSE 1.00 END
    * CASE WHEN (o.id + k) % 3 = 0 THEN 0.85 ELSE 1.00 END,
    2)
FROM orders o
CROSS JOIN generate_series(0, 2) AS k
JOIN product p ON p.id = ((o.id * 3 + k) % {PRODUCTS}) + 1
"""

# --------------------------------------------------------------------------
# Decoys
#
# Rows matter as much as the table exists: an empty table is dismissed in one
# tool call, a populated one has to be looked at. Several are deliberately
# tempting — app_user has the soft-delete column, cart mirrors orders, and
# invoice/payment look like revenue.
# --------------------------------------------------------------------------

DECOY_SQL = [
    """INSERT INTO app_user (email, display_name, created_at, deleted_at, last_login_at)
       SELECT 'user' || i || '@example.com', 'User ' || i,
              now() - ((i % 900) || ' days')::interval,
              CASE WHEN i % 17 = 0 THEN now() - ((i % 90) || ' days')::interval END,
              now() - ((i % 30) || ' days')::interval
       FROM generate_series(1, 900) AS i""",
    """INSERT INTO role (key, description)
       SELECT 'role_' || i, 'Role number ' || i FROM generate_series(1, 12) AS i""",
    """INSERT INTO permission (key, resource, action)
       SELECT 'perm_' || i, 'resource_' || (i % 15), (ARRAY['read','write','delete','admin'])[(i % 4) + 1]
       FROM generate_series(1, 60) AS i""",
    """INSERT INTO role_permission (role_id, permission_id, granted_at)
       SELECT (i % 12) + 1, (i % 60) + 1, now() - ((i % 400) || ' days')::interval
       FROM generate_series(1, 180) AS i""",
    """INSERT INTO user_role (user_id, role_id, assigned_at)
       SELECT (i % 900) + 1, (i % 12) + 1, now() - ((i % 400) || ' days')::interval
       FROM generate_series(1, 1200) AS i""",
    """INSERT INTO api_key (user_id, prefix, created_at, revoked_at)
       SELECT (i % 900) + 1, 'ak_' || lpad(i::text, 6, '0'),
              now() - ((i % 500) || ' days')::interval,
              CASE WHEN i % 9 = 0 THEN now() - ((i % 60) || ' days')::interval END
       FROM generate_series(1, 240) AS i""",
    """INSERT INTO session (user_id, token_hash, started_at, expires_at, ip)
       SELECT (i % 900) + 1, md5(i::text), now() - ((i % 60) || ' days')::interval,
              now() + ((i % 30) || ' days')::interval,
              ('10.0.' || (i % 255) || '.' || ((i * 7) % 255))::inet
       FROM generate_series(1, 3000) AS i""",
    """INSERT INTO audit_log (actor_id, action, entity_type, entity_id, at, payload)
       SELECT (i % 900) + 1, (ARRAY['create','update','delete','login'])[(i % 4) + 1],
              (ARRAY['order','customer','product','user'])[(i % 4) + 1], (i % 2000) + 1,
              now() - ((i % 700) || ' days')::interval, jsonb_build_object('seq', i)
       FROM generate_series(1, 5000) AS i""",
    """INSERT INTO feature_flag (key, enabled, rollout_pct, updated_at)
       SELECT 'flag_' || i, i % 3 = 0, (i * 7) % 101, now() - ((i % 200) || ' days')::interval
       FROM generate_series(1, 40) AS i""",
    """INSERT INTO notification (user_id, kind, body, read_at, created_at)
       SELECT (i % 900) + 1, (ARRAY['email','push','sms'])[(i % 3) + 1], 'Notification ' || i,
              CASE WHEN i % 4 <> 0 THEN now() - ((i % 20) || ' days')::interval END,
              now() - ((i % 200) || ' days')::interval
       FROM generate_series(1, 2400) AS i""",
    """INSERT INTO email_template (key, subject, locale)
       SELECT 'tpl_' || i, 'Subject line ' || i, (ARRAY['en','fr','de'])[(i % 3) + 1]
       FROM generate_series(1, 30) AS i""",
    """INSERT INTO email_queue (template_id, recipient, queued_at, sent_at, status)
       SELECT (i % 30) + 1, 'user' || i || '@example.com',
              now() - ((i % 90) || ' days')::interval,
              CASE WHEN i % 7 <> 0 THEN now() - ((i % 89) || ' days')::interval END,
              CASE WHEN i % 7 = 0 THEN 'pending' ELSE 'sent' END
       FROM generate_series(1, 1500) AS i""",
    """INSERT INTO webhook_delivery (endpoint_url, event_type, status_code, attempted_at, attempts)
       SELECT 'https://hooks.example.com/' || (i % 20), 'order.' || (ARRAY['created','paid','shipped'])[(i % 3) + 1],
              CASE WHEN i % 11 = 0 THEN 500 ELSE 200 END,
              now() - ((i % 120) || ' days')::interval, (i % 3) + 1
       FROM generate_series(1, 2000) AS i""",
    f"""INSERT INTO address (customer_id, line1, city, postcode, country_code, is_default)
       SELECT (i % {CUSTOMERS}) + 1, i || ' Example Street',
              (ARRAY['Leeds','Bristol','Denver','Lyon'])[(i % 4) + 1],
              'PC' || lpad(i::text, 5, '0'),
              (ARRAY['GB','US','FR','DE'])[(i % 4) + 1], i % 3 = 0
       FROM generate_series(1, 2600) AS i""",
    """INSERT INTO country (code, name, region, currency_code)
       SELECT chr(65 + (i / 26)) || chr(65 + (i % 26)), 'Country ' || i,
              (ARRAY['emea','amer','apac'])[(i % 3) + 1],
              (ARRAY['GBP','USD','EUR'])[(i % 3) + 1]
       FROM generate_series(0, 119) AS i""",
    """INSERT INTO currency (code, name, minor_unit)
       SELECT 'C' || lpad(i::text, 2, '0'), 'Currency ' || i, 2
       FROM generate_series(1, 40) AS i""",
    """INSERT INTO exchange_rate (base_code, quote_code, rate, as_of)
       SELECT 'GBP', 'C' || lpad(((i % 40) + 1)::text, 2, '0'),
              ROUND((0.5 + (i % 300) * 0.01)::numeric, 6),
              DATE '2024-01-01' + (i % 800)
       FROM generate_series(1, 3200) AS i""",
    """INSERT INTO tax_rate (country_code, rate, effective_from)
       SELECT (ARRAY['GB','US','FR','DE'])[(i % 4) + 1],
              ROUND((0.05 + (i % 15) * 0.01)::numeric, 4), DATE '2021-01-01' + (i * 30)
       FROM generate_series(1, 48) AS i""",
    """INSERT INTO shipping_method (name, carrier, base_cost, est_days)
       SELECT 'Method ' || i, (ARRAY['Royal Mail','DHL','UPS','DPD'])[(i % 4) + 1],
              ROUND((2 + i * 0.75)::numeric, 2), (i % 7) + 1
       FROM generate_series(1, 16) AS i""",
    f"""INSERT INTO shipment (order_id, method_id, tracking_ref, shipped_at, delivered_at)
       SELECT (i % {ORDERS}) + 1, (i % 16) + 1, 'TRK' || lpad(i::text, 9, '0'),
              now() - ((i % 600) || ' days')::interval,
              CASE WHEN i % 6 <> 0 THEN now() - ((i % 598) || ' days')::interval END
       FROM generate_series(1, 4200) AS i""",
    """INSERT INTO warehouse (code, name, country_code)
       SELECT 'WH' || lpad(i::text, 3, '0'), 'Warehouse ' || i,
              (ARRAY['GB','US','FR','DE'])[(i % 4) + 1]
       FROM generate_series(1, 14) AS i""",
    f"""INSERT INTO inventory_level (warehouse_id, product_id, on_hand, reserved, updated_at)
       SELECT (i % 14) + 1, (i % {PRODUCTS}) + 1, (i * 13) % 900, (i * 3) % 40,
              now() - ((i % 30) || ' days')::interval
       FROM generate_series(1, 2800) AS i""",
    f"""INSERT INTO stock_movement (warehouse_id, product_id, delta, reason, at)
       SELECT (i % 14) + 1, (i % {PRODUCTS}) + 1,
              CASE WHEN i % 2 = 0 THEN (i % 50) + 1 ELSE -((i % 30) + 1) END,
              (ARRAY['receipt','sale','adjustment','return'])[(i % 4) + 1],
              now() - ((i % 500) || ' days')::interval
       FROM generate_series(1, 6000) AS i""",
    """INSERT INTO supplier (name, country_code, active)
       SELECT 'Supplier ' || i, (ARRAY['GB','US','FR','DE'])[(i % 4) + 1], i % 8 <> 0
       FROM generate_series(1, 70) AS i""",
    """INSERT INTO purchase_order (supplier_id, placed_at, expected_at, status)
       SELECT (i % 70) + 1, now() - ((i % 500) || ' days')::interval,
              CURRENT_DATE + (i % 60),
              (ARRAY['draft','placed','received','cancelled'])[(i % 4) + 1]
       FROM generate_series(1, 800) AS i""",
    f"""INSERT INTO purchase_order_item (purchase_order_id, product_id, qty, cost)
       SELECT (i % 800) + 1, (i % {PRODUCTS}) + 1, (i % 60) + 1,
              ROUND((2 + (i % 90) * 0.9)::numeric, 2)
       FROM generate_series(1, 2400) AS i""",
    """INSERT INTO category (name, parent_id, slug)
       SELECT 'Category ' || i, CASE WHEN i > 8 THEN (i % 8) + 1 END, 'cat-' || i
       FROM generate_series(1, 48) AS i""",
    f"""INSERT INTO product_category (product_id, category_id, is_primary)
       SELECT (i % {PRODUCTS}) + 1, (i % 48) + 1, i % 3 = 0
       FROM generate_series(1, 520) AS i""",
    f"""INSERT INTO price_history (product_id, price, effective_from, effective_to)
       SELECT (i % {PRODUCTS}) + 1, ROUND((4 + (i % 200) * 1.1)::numeric, 2),
              DATE '2023-01-01' + (i % 900), DATE '2023-06-01' + (i % 900)
       FROM generate_series(1, 900) AS i""",
    """INSERT INTO discount_code (code, percent_off, valid_from, valid_to, max_uses)
       SELECT 'SAVE' || lpad(i::text, 4, '0'), (i % 40) + 5,
              DATE '2024-01-01' + (i % 700), DATE '2024-04-01' + (i % 700), (i % 500) + 10
       FROM generate_series(1, 260) AS i""",
    f"""INSERT INTO cart (customer_id, created_at, abandoned_at)
       SELECT (i % {CUSTOMERS}) + 1, now() - ((i % 200) || ' days')::interval,
              CASE WHEN i % 3 = 0 THEN now() - ((i % 199) || ' days')::interval END
       FROM generate_series(1, 1700) AS i""",
    f"""INSERT INTO cart_item (cart_id, product_id, qty, added_at)
       SELECT (i % 1700) + 1, (i % {PRODUCTS}) + 1, (i % 5) + 1,
              now() - ((i % 200) || ' days')::interval
       FROM generate_series(1, 4300) AS i""",
    f"""INSERT INTO payment (order_id, amount, method, captured_at, status)
       SELECT (i % {ORDERS}) + 1, ROUND((12 + (i % 900) * 1.4)::numeric, 2),
              (ARRAY['card','paypal','transfer'])[(i % 3) + 1],
              now() - ((i % 700) || ' days')::interval,
              CASE WHEN i % 13 = 0 THEN 'failed' ELSE 'captured' END
       FROM generate_series(1, 5400) AS i""",
    """INSERT INTO refund (payment_id, amount, reason, refunded_at)
       SELECT (i % 5400) + 1, ROUND((5 + (i % 200) * 1.1)::numeric, 2),
              (ARRAY['damaged','late','changed mind'])[(i % 3) + 1],
              now() - ((i % 400) || ' days')::interval
       FROM generate_series(1, 600) AS i""",
    f"""INSERT INTO invoice (order_id, number, total, issued_at, paid_at)
       SELECT (i % {ORDERS}) + 1, 'INV-' || lpad(i::text, 7, '0'),
              ROUND((20 + (i % 800) * 1.6)::numeric, 2),
              now() - ((i % 700) || ' days')::interval,
              CASE WHEN i % 9 <> 0 THEN now() - ((i % 690) || ' days')::interval END
       FROM generate_series(1, 5200) AS i""",
    f"""INSERT INTO support_ticket (customer_id, subject, status, priority, opened_at, closed_at)
       SELECT (i % {CUSTOMERS}) + 1, 'Ticket subject ' || i,
              (ARRAY['open','pending','closed'])[(i % 3) + 1],
              (ARRAY['low','normal','high','urgent'])[(i % 4) + 1],
              now() - ((i % 600) || ' days')::interval,
              CASE WHEN i % 3 = 2 THEN now() - ((i % 590) || ' days')::interval END
       FROM generate_series(1, 1400) AS i""",
]


async def main() -> None:
    started = time.monotonic()
    async with await psycopg.AsyncConnection.connect(
        settings().database_url, row_factory=dict_row
    ) as conn:
        # Idempotent: wipe what this script owns, leave the cache alone.
        await conn.execute(
            "TRUNCATE TABLE " + ", ".join(SEEDED) + " RESTART IDENTITY CASCADE"
        )

        for label, sql in [
            ("customer", CUSTOMER_SQL),
            ("product", PRODUCT_SQL),
            ("orders", ORDERS_SQL),
            ("order_item", ORDER_ITEM_SQL),
        ]:
            await conn.execute(sql)
            print(f"  seeded {label}")

        for sql in DECOY_SQL:
            await conn.execute(sql)
        print(f"  seeded {len(DECOY_SQL)} decoy tables")

        await conn.commit()

        cur = await conn.execute("""
            SELECT
              (SELECT count(*) FROM customer)                              AS customers,
              (SELECT count(*) FROM customer WHERE deleted_at IS NULL)     AS active,
              (SELECT count(*) FROM customer WHERE region = 'west')        AS naive_west,
              (SELECT count(*) FROM customer WHERE lower(region) = 'west') AS real_west,
              (SELECT count(*) FROM orders)                                AS orders,
              (SELECT count(*) FROM orders WHERE status = 'cancelled')     AS cancelled,
              (SELECT count(*) FROM order_item)                            AS items,
              (SELECT count(*) FROM information_schema.tables
                 WHERE table_schema = 'public')                            AS tables
        """)
        s = await cur.fetchone()
        assert s is not None

    print(
        f"\n  {s['tables']} tables, {s['customers']} customers "
        f"({s['active']} active), {s['orders']} orders "
        f"({s['cancelled']} cancelled), {s['items']} order items"
    )
    print(
        f"  region trap: WHERE region = 'west' finds {s['naive_west']}, "
        f"lower(region) finds {s['real_west']}"
    )
    print(f"seed complete in {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
