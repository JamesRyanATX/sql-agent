A query just ran successfully. Write down what a newcomer to this database would need to know to get it right first time.

You are given the SQL that actually executed. **Everything you record must be supported by that SQL or by the schema findings — not by what you meant to do.** If the SQL excludes two statuses, the recipe covers two statuses, whatever the intent was.

Record two kinds of thing:

- **schema_fact** — something durably true about the shape of the data: a table's purpose, a column that isn't named what you'd guess, a join key, an enum's real values.
- **recipe** — how to express a business concept in SQL. Give it the name a person would use ("revenue", "active customer") and a `sql_fragment` copied from the query that ran.

Neither kind is a census. A row count, a percentage, or a parenthetical like "(currently 1,840)" is this query's answer, not a fact about the schema — it goes stale the instant a row changes, and nothing ever revisits it to check. "deleted_at is a nullable soft-delete flag" is a schema_fact; "160 of 2,000 rows are soft-deleted" is not — write the rule, not the count it produced today.

Every entry needs a short, stable `name` — it is the key this is filed under, and reusing a name **overwrites** what is already there.

Reuse a name only when this query taught you more about *that same concept*, so the note is refined rather than duplicated. If what you learned is narrower, broader, or merely related — "revenue" against "revenue by region", "active customer" against "active customer in a region" — give it its own name. Overwriting a general rule with a special case destroys the general rule, and every later question that relied on it inherits the narrower one.

Write claims as plain English a colleague could read aloud. State the convention, not the query you happened to write: "an active customer is one whose deleted_at is null", not "I filtered on deleted_at".

Record only what this query actually establishes. Nothing speculative, nothing you did not verify, and nothing that merely restates the question.
