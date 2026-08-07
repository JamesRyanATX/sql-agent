-- The demo database, whole: role, schema, and data (PLAN.md §3.2).
--
--     make seed
--
-- This is not the product. It is a booby-trapped fixture that exists so the
-- agent has something to explore — which is why it lives here next to the tape
-- that records it, and not in `migrations/`. `migrations/` is the agent's own
-- state, on a different server.
--
-- Two properties matter. Wide, not deep: ~40 tables of which 4 are relevant,
-- so exploration has to genuinely search. And booby-trapped, so exploration
-- produces recipes worth reading aloud.
--
-- **Deterministic.** Every value derives from `generate_series` via modular
-- arithmetic, never `random()`, so *1,840 active customers is a fact, not a
-- probability* — it is the answer beat 1 of the demo is gated on, and it has to
-- be the same number every rehearsal. The report at the bottom is what proves
-- it still is.
--
-- **Idempotent.** CREATE ... IF NOT EXISTS, a guarded role, and a TRUNCATE
-- before every insert. Running it twice is running it once.
--
-- **No psql meta-commands** (\echo, \i, \set). tests/conftest.py applies this
-- file through psycopg, which cannot parse them. That is what the report at the
-- bottom replaces.


-- ===========================================================================
-- The read-only role the agent connects as
--
-- The agent reaches this database through `reader`, which holds SELECT and
-- nothing else. db.target_readonly()'s transaction guard is the second line of
-- defence now, not the only one.
-- ===========================================================================

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'reader') THEN
    CREATE ROLE reader LOGIN PASSWORD 'reader';
  END IF;
END
$$;

-- GRANT CONNECT takes no expression, so naming the database here would break
-- the test database, which is called something else.
DO $$
BEGIN
  EXECUTE format('GRANT CONNECT ON DATABASE %I TO reader', current_database());
END
$$;

GRANT USAGE ON SCHEMA public TO reader;

-- Covers the tables created below. The blanket grant at the bottom covers
-- re-runs, and anything that existed before this line did.
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO reader;


-- ===========================================================================
-- The four tables that matter
-- ===========================================================================

CREATE TABLE IF NOT EXISTS customer (
  id         bigserial PRIMARY KEY,
  name       text NOT NULL,
  region     text,          -- TRAP 2: casing is not normalised
  signed_up  date NOT NULL,
  deleted_at timestamptz    -- TRAP 1: soft deletes, ~8% of rows
);

CREATE TABLE IF NOT EXISTS product (
  id           bigserial PRIMARY KEY,
  name         text NOT NULL,
  unit_price   numeric(10, 2) NOT NULL,  -- TRAP 4: *current* price, not historical
  discontinued boolean NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS orders (
  id          bigserial PRIMARY KEY,
  customer_id bigint NOT NULL REFERENCES customer (id),
  created     timestamptz NOT NULL,  -- TRAP 5: `created`, never `created_at`
  status      text NOT NULL          -- TRAP 3: includes 'cancelled'
);

CREATE TABLE IF NOT EXISTS order_item (
  order_id   bigint NOT NULL REFERENCES orders (id),
  product_id bigint NOT NULL REFERENCES product (id),
  qty        int NOT NULL,
  price      numeric(10, 2) NOT NULL,  -- TRAP 4: price *at time of sale*
  PRIMARY KEY (order_id, product_id)
);

CREATE INDEX IF NOT EXISTS orders_customer_idx ON orders (customer_id);
CREATE INDEX IF NOT EXISTS orders_created_idx ON orders (created);


-- ===========================================================================
-- ~36 decoys
--
-- Plausible warehouse noise. Several are deliberately tempting: `app_user` has
-- the soft-delete column you were looking for but is not the customer table,
-- `cart`/`cart_item` mirror the orders shape, `invoice` and `payment` look
-- like revenue, and `price_history` looks like the answer to the price trap.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS app_user (
  id bigserial PRIMARY KEY, email text, display_name text,
  created_at timestamptz, deleted_at timestamptz, last_login_at timestamptz);

CREATE TABLE IF NOT EXISTS role (
  id bigserial PRIMARY KEY, key text, description text);

CREATE TABLE IF NOT EXISTS permission (
  id bigserial PRIMARY KEY, key text, resource text, action text);

CREATE TABLE IF NOT EXISTS role_permission (
  role_id bigint, permission_id bigint, granted_at timestamptz);

CREATE TABLE IF NOT EXISTS user_role (
  user_id bigint, role_id bigint, assigned_at timestamptz);

CREATE TABLE IF NOT EXISTS api_key (
  id bigserial PRIMARY KEY, user_id bigint, prefix text,
  created_at timestamptz, revoked_at timestamptz);

CREATE TABLE IF NOT EXISTS session (
  id bigserial PRIMARY KEY, user_id bigint, token_hash text,
  started_at timestamptz, expires_at timestamptz, ip inet);

CREATE TABLE IF NOT EXISTS audit_log (
  id bigserial PRIMARY KEY, actor_id bigint, action text,
  entity_type text, entity_id bigint, at timestamptz, payload jsonb);

CREATE TABLE IF NOT EXISTS feature_flag (
  id bigserial PRIMARY KEY, key text, enabled boolean,
  rollout_pct int, updated_at timestamptz);

CREATE TABLE IF NOT EXISTS notification (
  id bigserial PRIMARY KEY, user_id bigint, kind text,
  body text, read_at timestamptz, created_at timestamptz);

CREATE TABLE IF NOT EXISTS email_template (
  id bigserial PRIMARY KEY, key text, subject text, locale text);

CREATE TABLE IF NOT EXISTS email_queue (
  id bigserial PRIMARY KEY, template_id bigint, recipient text,
  queued_at timestamptz, sent_at timestamptz, status text);

CREATE TABLE IF NOT EXISTS webhook_delivery (
  id bigserial PRIMARY KEY, endpoint_url text, event_type text,
  status_code int, attempted_at timestamptz, attempts int);

CREATE TABLE IF NOT EXISTS address (
  id bigserial PRIMARY KEY, customer_id bigint, line1 text,
  city text, postcode text, country_code text, is_default boolean);

CREATE TABLE IF NOT EXISTS country (
  code text PRIMARY KEY, name text, region text, currency_code text);

CREATE TABLE IF NOT EXISTS currency (
  code text PRIMARY KEY, name text, minor_unit int);

CREATE TABLE IF NOT EXISTS exchange_rate (
  id bigserial PRIMARY KEY, base_code text, quote_code text,
  rate numeric(14, 6), as_of date);

CREATE TABLE IF NOT EXISTS tax_rate (
  id bigserial PRIMARY KEY, country_code text, rate numeric(5, 4),
  effective_from date);

CREATE TABLE IF NOT EXISTS shipping_method (
  id bigserial PRIMARY KEY, name text, carrier text,
  base_cost numeric(10, 2), est_days int);

CREATE TABLE IF NOT EXISTS shipment (
  id bigserial PRIMARY KEY, order_id bigint, method_id bigint,
  tracking_ref text, shipped_at timestamptz, delivered_at timestamptz);

CREATE TABLE IF NOT EXISTS warehouse (
  id bigserial PRIMARY KEY, code text, name text, country_code text);

CREATE TABLE IF NOT EXISTS inventory_level (
  warehouse_id bigint, product_id bigint, on_hand int,
  reserved int, updated_at timestamptz);

CREATE TABLE IF NOT EXISTS stock_movement (
  id bigserial PRIMARY KEY, warehouse_id bigint, product_id bigint,
  delta int, reason text, at timestamptz);

CREATE TABLE IF NOT EXISTS supplier (
  id bigserial PRIMARY KEY, name text, country_code text, active boolean);

CREATE TABLE IF NOT EXISTS purchase_order (
  id bigserial PRIMARY KEY, supplier_id bigint, placed_at timestamptz,
  expected_at date, status text);

CREATE TABLE IF NOT EXISTS purchase_order_item (
  purchase_order_id bigint, product_id bigint, qty int,
  cost numeric(10, 2));

CREATE TABLE IF NOT EXISTS category (
  id bigserial PRIMARY KEY, name text, parent_id bigint, slug text);

CREATE TABLE IF NOT EXISTS product_category (
  product_id bigint, category_id bigint, is_primary boolean);

CREATE TABLE IF NOT EXISTS price_history (
  id bigserial PRIMARY KEY, product_id bigint, price numeric(10, 2),
  effective_from date, effective_to date);

CREATE TABLE IF NOT EXISTS discount_code (
  id bigserial PRIMARY KEY, code text, percent_off int,
  valid_from date, valid_to date, max_uses int);

CREATE TABLE IF NOT EXISTS cart (
  id bigserial PRIMARY KEY, customer_id bigint,
  created_at timestamptz, abandoned_at timestamptz);

CREATE TABLE IF NOT EXISTS cart_item (
  cart_id bigint, product_id bigint, qty int, added_at timestamptz);

CREATE TABLE IF NOT EXISTS payment (
  id bigserial PRIMARY KEY, order_id bigint, amount numeric(12, 2),
  method text, captured_at timestamptz, status text);

CREATE TABLE IF NOT EXISTS refund (
  id bigserial PRIMARY KEY, payment_id bigint, amount numeric(12, 2),
  reason text, refunded_at timestamptz);

CREATE TABLE IF NOT EXISTS invoice (
  id bigserial PRIMARY KEY, order_id bigint, number text,
  total numeric(12, 2), issued_at timestamptz, paid_at timestamptz);

CREATE TABLE IF NOT EXISTS support_ticket (
  id bigserial PRIMARY KEY, customer_id bigint, subject text,
  status text, priority text, opened_at timestamptz, closed_at timestamptz);


-- ===========================================================================
-- Data
--
-- Everything below is deterministic. Wiped first, so this file is safe to
-- re-apply — that is what `make reset` leans on.
-- ===========================================================================

TRUNCATE TABLE
  order_item, orders, product, customer,
  app_user, role, permission, role_permission, user_role,
  api_key, session, audit_log, feature_flag, notification,
  email_template, email_queue, webhook_delivery, address,
  country, currency, exchange_rate, tax_rate, shipping_method,
  shipment, warehouse, inventory_level, stock_movement, supplier,
  purchase_order, purchase_order_item, category, product_category,
  price_history, discount_code, cart, cart_item, payment,
  refund, invoice, support_ticket
RESTART IDENTITY CASCADE;


-- --------------------------------------------------------------- customer ---

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

  -- TRAP 1. Soft deletes on exactly 160 of 2000 rows, so the honest answer to
  -- "how many customers do we have?" is 1,840 and the obvious SELECT count(*)
  -- is wrong by 160.
  CASE WHEN i % 25 IN (0, 1)
       THEN TIMESTAMPTZ '2024-06-01' + ((i % 400) || ' days')::interval
  END
FROM generate_series(1, 2000) AS i;


-- ---------------------------------------------------------------- product ---

INSERT INTO product (name, unit_price, discontinued)
SELECT
  'Product ' || i,
  -- TRAP 4, half of it: this is the price *now*. It is not what anything
  -- was actually sold for.
  ROUND((5 + (i % 180) * 1.37)::numeric, 2),
  i % 11 = 0
FROM generate_series(1, 200) AS i;


-- ----------------------------------------------------------------- orders ---

INSERT INTO orders (customer_id, created, status)
SELECT
  ((i * 7) % 2000) + 1,

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
FROM generate_series(1, 6000) AS i;


-- ------------------------------------------------------------- order_item ---

INSERT INTO order_item (order_id, product_id, qty, price)
SELECT
  o.id,
  ((o.id * 3 + k) % 200) + 1,
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
JOIN product p ON p.id = ((o.id * 3 + k) % 200) + 1;


-- ------------------------------------------------------------------ decoys ---
--
-- Rows matter as much as the table existing: an empty table is dismissed in one
-- tool call, a populated one has to be looked at.

INSERT INTO app_user (email, display_name, created_at, deleted_at, last_login_at)
SELECT 'user' || i || '@example.com', 'User ' || i,
       now() - ((i % 900) || ' days')::interval,
       CASE WHEN i % 17 = 0 THEN now() - ((i % 90) || ' days')::interval END,
       now() - ((i % 30) || ' days')::interval
FROM generate_series(1, 900) AS i;

INSERT INTO role (key, description)
SELECT 'role_' || i, 'Role number ' || i FROM generate_series(1, 12) AS i;

INSERT INTO permission (key, resource, action)
SELECT 'perm_' || i, 'resource_' || (i % 15),
       (ARRAY['read','write','delete','admin'])[(i % 4) + 1]
FROM generate_series(1, 60) AS i;

INSERT INTO role_permission (role_id, permission_id, granted_at)
SELECT (i % 12) + 1, (i % 60) + 1, now() - ((i % 400) || ' days')::interval
FROM generate_series(1, 180) AS i;

INSERT INTO user_role (user_id, role_id, assigned_at)
SELECT (i % 900) + 1, (i % 12) + 1, now() - ((i % 400) || ' days')::interval
FROM generate_series(1, 1200) AS i;

INSERT INTO api_key (user_id, prefix, created_at, revoked_at)
SELECT (i % 900) + 1, 'ak_' || lpad(i::text, 6, '0'),
       now() - ((i % 500) || ' days')::interval,
       CASE WHEN i % 9 = 0 THEN now() - ((i % 60) || ' days')::interval END
FROM generate_series(1, 240) AS i;

INSERT INTO session (user_id, token_hash, started_at, expires_at, ip)
SELECT (i % 900) + 1, md5(i::text), now() - ((i % 60) || ' days')::interval,
       now() + ((i % 30) || ' days')::interval,
       ('10.0.' || (i % 255) || '.' || ((i * 7) % 255))::inet
FROM generate_series(1, 3000) AS i;

INSERT INTO audit_log (actor_id, action, entity_type, entity_id, at, payload)
SELECT (i % 900) + 1, (ARRAY['create','update','delete','login'])[(i % 4) + 1],
       (ARRAY['order','customer','product','user'])[(i % 4) + 1], (i % 2000) + 1,
       now() - ((i % 700) || ' days')::interval, jsonb_build_object('seq', i)
FROM generate_series(1, 5000) AS i;

INSERT INTO feature_flag (key, enabled, rollout_pct, updated_at)
SELECT 'flag_' || i, i % 3 = 0, (i * 7) % 101,
       now() - ((i % 200) || ' days')::interval
FROM generate_series(1, 40) AS i;

INSERT INTO notification (user_id, kind, body, read_at, created_at)
SELECT (i % 900) + 1, (ARRAY['email','push','sms'])[(i % 3) + 1],
       'Notification ' || i,
       CASE WHEN i % 4 <> 0 THEN now() - ((i % 20) || ' days')::interval END,
       now() - ((i % 200) || ' days')::interval
FROM generate_series(1, 2400) AS i;

INSERT INTO email_template (key, subject, locale)
SELECT 'tpl_' || i, 'Subject line ' || i, (ARRAY['en','fr','de'])[(i % 3) + 1]
FROM generate_series(1, 30) AS i;

INSERT INTO email_queue (template_id, recipient, queued_at, sent_at, status)
SELECT (i % 30) + 1, 'user' || i || '@example.com',
       now() - ((i % 90) || ' days')::interval,
       CASE WHEN i % 7 <> 0 THEN now() - ((i % 89) || ' days')::interval END,
       CASE WHEN i % 7 = 0 THEN 'pending' ELSE 'sent' END
FROM generate_series(1, 1500) AS i;

INSERT INTO webhook_delivery (endpoint_url, event_type, status_code, attempted_at, attempts)
SELECT 'https://hooks.example.com/' || (i % 20),
       'order.' || (ARRAY['created','paid','shipped'])[(i % 3) + 1],
       CASE WHEN i % 11 = 0 THEN 500 ELSE 200 END,
       now() - ((i % 120) || ' days')::interval, (i % 3) + 1
FROM generate_series(1, 2000) AS i;

INSERT INTO address (customer_id, line1, city, postcode, country_code, is_default)
SELECT (i % 2000) + 1, i || ' Example Street',
       (ARRAY['Leeds','Bristol','Denver','Lyon'])[(i % 4) + 1],
       'PC' || lpad(i::text, 5, '0'),
       (ARRAY['GB','US','FR','DE'])[(i % 4) + 1], i % 3 = 0
FROM generate_series(1, 2600) AS i;

INSERT INTO country (code, name, region, currency_code)
SELECT chr(65 + (i / 26)) || chr(65 + (i % 26)), 'Country ' || i,
       (ARRAY['emea','amer','apac'])[(i % 3) + 1],
       (ARRAY['GBP','USD','EUR'])[(i % 3) + 1]
FROM generate_series(0, 119) AS i;

INSERT INTO currency (code, name, minor_unit)
SELECT 'C' || lpad(i::text, 2, '0'), 'Currency ' || i, 2
FROM generate_series(1, 40) AS i;

INSERT INTO exchange_rate (base_code, quote_code, rate, as_of)
SELECT 'GBP', 'C' || lpad(((i % 40) + 1)::text, 2, '0'),
       ROUND((0.5 + (i % 300) * 0.01)::numeric, 6),
       DATE '2024-01-01' + (i % 800)
FROM generate_series(1, 3200) AS i;

INSERT INTO tax_rate (country_code, rate, effective_from)
SELECT (ARRAY['GB','US','FR','DE'])[(i % 4) + 1],
       ROUND((0.05 + (i % 15) * 0.01)::numeric, 4), DATE '2021-01-01' + (i * 30)
FROM generate_series(1, 48) AS i;

INSERT INTO shipping_method (name, carrier, base_cost, est_days)
SELECT 'Method ' || i, (ARRAY['Royal Mail','DHL','UPS','DPD'])[(i % 4) + 1],
       ROUND((2 + i * 0.75)::numeric, 2), (i % 7) + 1
FROM generate_series(1, 16) AS i;

INSERT INTO shipment (order_id, method_id, tracking_ref, shipped_at, delivered_at)
SELECT (i % 6000) + 1, (i % 16) + 1, 'TRK' || lpad(i::text, 9, '0'),
       now() - ((i % 600) || ' days')::interval,
       CASE WHEN i % 6 <> 0 THEN now() - ((i % 598) || ' days')::interval END
FROM generate_series(1, 4200) AS i;

INSERT INTO warehouse (code, name, country_code)
SELECT 'WH' || lpad(i::text, 3, '0'), 'Warehouse ' || i,
       (ARRAY['GB','US','FR','DE'])[(i % 4) + 1]
FROM generate_series(1, 14) AS i;

INSERT INTO inventory_level (warehouse_id, product_id, on_hand, reserved, updated_at)
SELECT (i % 14) + 1, (i % 200) + 1, (i * 13) % 900, (i * 3) % 40,
       now() - ((i % 30) || ' days')::interval
FROM generate_series(1, 2800) AS i;

INSERT INTO stock_movement (warehouse_id, product_id, delta, reason, at)
SELECT (i % 14) + 1, (i % 200) + 1,
       CASE WHEN i % 2 = 0 THEN (i % 50) + 1 ELSE -((i % 30) + 1) END,
       (ARRAY['receipt','sale','adjustment','return'])[(i % 4) + 1],
       now() - ((i % 500) || ' days')::interval
FROM generate_series(1, 6000) AS i;

INSERT INTO supplier (name, country_code, active)
SELECT 'Supplier ' || i, (ARRAY['GB','US','FR','DE'])[(i % 4) + 1], i % 8 <> 0
FROM generate_series(1, 70) AS i;

INSERT INTO purchase_order (supplier_id, placed_at, expected_at, status)
SELECT (i % 70) + 1, now() - ((i % 500) || ' days')::interval,
       CURRENT_DATE + (i % 60),
       (ARRAY['draft','placed','received','cancelled'])[(i % 4) + 1]
FROM generate_series(1, 800) AS i;

INSERT INTO purchase_order_item (purchase_order_id, product_id, qty, cost)
SELECT (i % 800) + 1, (i % 200) + 1, (i % 60) + 1,
       ROUND((2 + (i % 90) * 0.9)::numeric, 2)
FROM generate_series(1, 2400) AS i;

INSERT INTO category (name, parent_id, slug)
SELECT 'Category ' || i, CASE WHEN i > 8 THEN (i % 8) + 1 END, 'cat-' || i
FROM generate_series(1, 48) AS i;

INSERT INTO product_category (product_id, category_id, is_primary)
SELECT (i % 200) + 1, (i % 48) + 1, i % 3 = 0
FROM generate_series(1, 520) AS i;

INSERT INTO price_history (product_id, price, effective_from, effective_to)
SELECT (i % 200) + 1, ROUND((4 + (i % 200) * 1.1)::numeric, 2),
       DATE '2023-01-01' + (i % 900), DATE '2023-06-01' + (i % 900)
FROM generate_series(1, 900) AS i;

INSERT INTO discount_code (code, percent_off, valid_from, valid_to, max_uses)
SELECT 'SAVE' || lpad(i::text, 4, '0'), (i % 40) + 5,
       DATE '2024-01-01' + (i % 700), DATE '2024-04-01' + (i % 700), (i % 500) + 10
FROM generate_series(1, 260) AS i;

INSERT INTO cart (customer_id, created_at, abandoned_at)
SELECT (i % 2000) + 1, now() - ((i % 200) || ' days')::interval,
       CASE WHEN i % 3 = 0 THEN now() - ((i % 199) || ' days')::interval END
FROM generate_series(1, 1700) AS i;

INSERT INTO cart_item (cart_id, product_id, qty, added_at)
SELECT (i % 1700) + 1, (i % 200) + 1, (i % 5) + 1,
       now() - ((i % 200) || ' days')::interval
FROM generate_series(1, 4300) AS i;

INSERT INTO payment (order_id, amount, method, captured_at, status)
SELECT (i % 6000) + 1, ROUND((12 + (i % 900) * 1.4)::numeric, 2),
       (ARRAY['card','paypal','transfer'])[(i % 3) + 1],
       now() - ((i % 700) || ' days')::interval,
       CASE WHEN i % 13 = 0 THEN 'failed' ELSE 'captured' END
FROM generate_series(1, 5400) AS i;

INSERT INTO refund (payment_id, amount, reason, refunded_at)
SELECT (i % 5400) + 1, ROUND((5 + (i % 200) * 1.1)::numeric, 2),
       (ARRAY['damaged','late','changed mind'])[(i % 3) + 1],
       now() - ((i % 400) || ' days')::interval
FROM generate_series(1, 600) AS i;

INSERT INTO invoice (order_id, number, total, issued_at, paid_at)
SELECT (i % 6000) + 1, 'INV-' || lpad(i::text, 7, '0'),
       ROUND((20 + (i % 800) * 1.6)::numeric, 2),
       now() - ((i % 700) || ' days')::interval,
       CASE WHEN i % 9 <> 0 THEN now() - ((i % 690) || ' days')::interval END
FROM generate_series(1, 5200) AS i;

INSERT INTO support_ticket (customer_id, subject, status, priority, opened_at, closed_at)
SELECT (i % 2000) + 1, 'Ticket subject ' || i,
       (ARRAY['open','pending','closed'])[(i % 3) + 1],
       (ARRAY['low','normal','high','urgent'])[(i % 4) + 1],
       now() - ((i % 600) || ' days')::interval,
       CASE WHEN i % 3 = 2 THEN now() - ((i % 590) || ' days')::interval END
FROM generate_series(1, 1400) AS i;


-- ===========================================================================
-- Grants, then the report
-- ===========================================================================

-- Belt and braces: the default privileges above cover tables created after
-- they were set, this covers everything else and every re-run.
GRANT SELECT ON ALL TABLES IN SCHEMA public TO reader;

-- What `scripts/seed.py` used to print. These numbers are the demo's contract —
-- 1,840 active and the 350-vs-500 region gap are what beats 1 and 3 are gated
-- on, and a translation that quietly changed them would be found on stage.
SELECT
  (SELECT count(*) FROM customer)                              AS customers,
  (SELECT count(*) FROM customer WHERE deleted_at IS NULL)     AS active,
  (SELECT count(*) FROM customer WHERE region = 'west')        AS naive_west,
  (SELECT count(*) FROM customer WHERE lower(region) = 'west') AS real_west,
  (SELECT count(*) FROM orders)                                AS orders,
  (SELECT count(*) FROM orders WHERE status = 'cancelled')     AS cancelled,
  (SELECT count(*) FROM order_item)                            AS items,
  (SELECT count(*) FROM information_schema.tables
     WHERE table_schema = 'public')                            AS tables;
