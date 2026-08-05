-- The business schema the agent has never seen (PLAN.md §3.2).
--
-- Two properties matter. Wide, not deep: ~40 tables of which 4 are relevant,
-- so exploration has to genuinely search. And booby-trapped, so exploration
-- produces recipes worth reading aloud.
--
-- Idempotent — `make migrate` re-applies every file on every run.

-- ===========================================================================
-- The four that matter
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
