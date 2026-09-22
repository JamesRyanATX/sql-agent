## list_tables

List every table in the database with its column count. Start here when you don't yet know what exists.

## describe_table

Show a table's columns, types, nullability, row count, primary key and foreign keys. Call this before writing SQL against a table — column names are frequently not what you'd guess.

## sample_column

Show distinct values from a column, plus how many rows are null. Call this when a column's meaning affects the answer — how many rows are populated is often the difference between a column that exists and a convention the data actually follows.

## count_distinct

Use when the user asks for a frequency breakdown of a column—how many rows have each distinct value, especially to understand a categorical column. Do not use it for a total row count or for counting rows matching a known condition, including a boolean or other yes/no value; use filtering and ordinary aggregation instead.
