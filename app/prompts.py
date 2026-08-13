"""The durable instruction blocks, and the seam an optimiser writes through.

Six strings, moved out of `graph.py` unchanged. What lives here is prose that
is a function of *nothing* — not the question, not the cache, not the dialect.
`render_cache()` and `dialect_note()` stay in `graph.py`, because they compose
per turn and are the content this prose tells the model what to do with. The
JSON schemas stay there too: they are the node's wire contract, not prose.

A module rather than six constants and an `if`, for two reasons.

`optim/` needs the seed candidate without importing the world. `from app.graph
import EXTRACT_SYSTEM` drags in langgraph, sqlalchemy, `app.db`, `app.tools`
and `app.store`; this module imports stdlib and `app.settings`.

And the override has to resolve **once per process**. `graph.plan` puts its
system block behind an Anthropic cache breakpoint on the promise that the block
varies with `connection_id` alone (§7.1). A prompt re-read per turn could change
between two turns of one server's life, and the only symptom would be T2 quietly
costing more. `_loaded` is memoised for that reason, not for speed — a candidate
that is constant for a whole process is strictly stronger than the invariant
asks for, which is why an override is safe here and would not be in `TurnState`
or in `config["configurable"]`.

Keys are the `node=` labels `llm.complete` records as the Langfuse generation
name, so a harvested trace and an override file name the same thing. `explore`
covers `explore.summary` as well: one prompt, two calls.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from app.settings import settings


# ------------------------------------------------------------------- the prose

EXPLORE_SYSTEM = """\
You are answering a business question against a SQL database you have \
never seen. Use the introspection tools to find out what you need.

The schema is wide and mostly irrelevant — expect to discard most tables. \
Column names are frequently not what you would guess, and a column existing \
tells you less than how its values are actually distributed.

Work until you could write correct SQL, then stop calling tools and reply with \
a short plain-English summary of what you found: the tables that matter, the \
join keys, and any convention the data follows that a newcomer would miss. \
Write it for someone who has not seen the tool output.

Deliver what was asked, at the scope intended. Do not explore tables that \
cannot affect this question."""

PLAN_SYSTEM = """\
You know some things about this database already. Decide whether they are \
enough to answer the question, and if they are, write the SQL now.

Enough means you can name every table and column the query needs and every \
convention that changes the result. A recipe whose SQL is marked unverified has \
never been run as written — it is a lead, not a fact.

If it is enough, set sufficient to true and return the SQL, listing the entries \
you relied on by name. Otherwise set it to false and say what you still need to \
find out — be specific, since that list is what the next step goes looking for. \
Guessing a column name is not sufficiency.

Answer exactly the question asked. Do not narrow it to something you have a \
recipe for, and do not broaden it — a cached recipe for a related question is a \
reason to say what is missing, not to answer a different question."""

SQL_SYSTEM = """\
Write one SELECT that answers the question, using the findings given.

Return the SQL and the assumptions you made — an assumption is anything a \
reader would need to know to agree the answer is correct, such as which rows \
you excluded and why. Read-only: no DDL, no writes, no CTEs that modify."""

FIX_SYSTEM = """\
The SQL failed. Given the error and the schema findings, return corrected SQL.

Fix the specific cause. Do not restructure the query beyond what the error \
requires."""

EXTRACT_SYSTEM = """\
A query just ran successfully. Write down what a newcomer to this database \
would need to know to get it right first time.

You are given the SQL that actually executed. **Everything you record must be \
supported by that SQL or by the schema findings — not by what you meant to do.** \
If the SQL excludes two statuses, the recipe covers two statuses, whatever the \
intent was.

Record two kinds of thing:

- **schema_fact** — something durably true about the shape of the data: a \
table's purpose, a column that isn't named what you'd guess, a join key, an \
enum's real values.
- **recipe** — how to express a business concept in SQL. Give it the name a \
person would use ("revenue", "active customer") and a `sql_fragment` copied \
from the query that ran.

Neither kind is a census. A row count, a percentage, or a parenthetical like \
"(currently 1,840)" is this query's answer, not a fact about the schema — it \
goes stale the instant a row changes, and nothing ever revisits it to check. \
"deleted_at is a nullable soft-delete flag" is a schema_fact; "160 of 2,000 \
rows are soft-deleted" is not — write the rule, not the count it produced today.

Every entry needs a short, stable `name` — it is the key this is filed under, \
and reusing a name **overwrites** what is already there.

Reuse a name only when this query taught you more about *that same concept*, so \
the note is refined rather than duplicated. If what you learned is narrower, \
broader, or merely related — "revenue" against "revenue by region", "active \
customer" against "active customer in a region" — give it its own name. \
Overwriting a general rule with a special case destroys the general rule, and \
every later question that relied on it inherits the narrower one.

Write claims as plain English a colleague could read aloud. State the \
convention, not the query you happened to write: "an active customer is one \
whose deleted_at is null", not "I filtered on deleted_at".

Record only what this query actually establishes. Nothing speculative, nothing \
you did not verify, and nothing that merely restates the question."""

# This used to say "for someone who did not see the query", and that reader
# existed: the whole result was `=> [(1840)]` and the answer text was not printed
# at all. Now the rows are a table directly above this, and a model told to "lead
# with the number" reads a nine-row result as nine numbers and types the table
# out again underneath itself.
#
# Measured on the quarters question: 353 output tokens before, 278 after, so the
# saving is real but modest — most of that 353 was thinking, not the recitation.
# The reason to change it is the duplication, not the tokens. What a table cannot
# say is what was counted and what the shape means, and that is the whole job
# left for prose.
ANSWER_SYSTEM = """\
State the answer in one or two sentences. The reader is looking at the result \
rows already, so do not list them back.

Say what the numbers mean and what was counted: the load-bearing assumptions \
inline in the prose, and anything about the shape of the result a reader would \
otherwise misread. Where the result is a single value, lead with that value. \
Do not restate the SQL, and do not add caveats that change nothing."""


# -------------------------------------------------------------------- the seam

_BLOCKS = {
    "explore": EXPLORE_SYSTEM,
    "plan": PLAN_SYSTEM,
    "generate_sql": SQL_SYSTEM,
    "fix": FIX_SYSTEM,
    "extract": EXTRACT_SYSTEM,
    "answer": ANSWER_SYSTEM,
}


@lru_cache
def _loaded() -> dict[str, str]:
    """The blocks above, with `prompt_dir` overrides applied.

    Memoised deliberately — see the module docstring. Tests and the optimiser
    call `_loaded.cache_clear()` after moving the directory, the same way they
    already call `settings.cache_clear()`.

    A `.txt` naming no prompt is an error rather than a no-op. A typo'd
    `extrct.txt` that silently changes nothing means an optimisation run that
    measures the seed and reports it as a candidate improvement, which is the
    kind of wrong that agrees with itself.
    """
    blocks = dict(_BLOCKS)
    configured = settings().prompt_dir
    if not configured:
        return blocks

    root = Path(configured)
    if not root.is_dir():
        raise ValueError(f"prompt_dir is set to {root}, which is not a directory")

    unknown = {p.name for p in root.glob("*.txt")} - {f"{n}.txt" for n in blocks}
    if unknown:
        raise ValueError(
            f"{root} holds .txt files naming no prompt: {sorted(unknown)} — "
            f"expected some of {sorted(f'{n}.txt' for n in blocks)}"
        )

    for name in blocks:
        override = root / f"{name}.txt"
        if override.is_file():
            blocks[name] = override.read_text(encoding="utf-8").strip()
    return blocks


def get(name: str) -> str:
    """The instruction block for a node, override applied."""
    return _loaded()[name]


def fingerprint() -> dict[str, str]:
    """node -> 8 hex chars of its prompt, for the turn span's metadata.

    The one thing Langfuse's own prompt management would have given for free:
    which prose produced this trace. Without it a harvest cannot tell a run
    under the seed from a run under a candidate, and round two of an
    optimisation trains on round one's output. Eight chars because this
    distinguishes revisions, it does not authenticate them.
    """
    return {
        name: hashlib.sha256(text.encode()).hexdigest()[:8]
        for name, text in sorted(_loaded().items())
    }
