A query just ran successfully. Write notes that will save work on the next question.

Follow these rules:
1. Counting questions ("how many ..."): save the number the query returned in the note, so the count never has to be run again.
2. Revenue and other totals: in the recipe's sql_fragment give only the aggregate expression, for example `sum(qty * price)`. Leave out FROM, JOIN and WHERE.
3. Queries that join tables: do not paste the join. Explain in words in the sql_fragment which tables are linked.
4. Breakdowns by a column (GROUP BY): file one note for each group, and give them all the same name, the name of the column.
