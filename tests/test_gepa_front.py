"""A GEPA report read back for a person.

The search prints its report to stderr once and writes it to a file. The file
is what the talk points at, so the display has to come back off the file: the
sentence saying what the score means, every candidate against every
validation example, how each eligible candidate opens, and the held-out table
for an overfit run's file. Each section is headed and starts at column one, so
there is a left edge for the eye.

Pure: a dict in, lines out, plus `main` over files written to a temporary
directory.
"""

from __future__ import annotations

import json

from tools.gepa import front

DOCUMENT = {
    "target": "extract",
    "score": "the fraction of recipes that quote the SQL that ran",
    "examples": ["first case", "second case", "third case"],
    "pool": {"candidates": 3, "eligible": 2, "metric_calls": 40, "seed_index": 0},
    "candidates": [
        {
            "index": 0, "seed": True, "val": 0.667,
            "scores": [1.0, 1.0, 0.0],
            "best_on": [1, 2], "chars": 2076,
            "components": {"extract": "the seed prompt"},
        },
        {
            "index": 1, "seed": False, "val": 0.333,
            "scores": [0.0, 1.0, None],
            "best_on": [], "chars": 1900,
            "components": {"extract": "a prompt nobody kept"},
        },
        {
            "index": 2, "seed": False, "val": 0.833,
            "scores": [0.5, 1.0, 1.0],
            "best_on": [2, 3], "chars": 3404,
            "components": {"extract": "the longer prompt\nsecond line"},
        },
    ],
}


HEADINGS = {"ABOUT", "SCORE", "SCORES", "HELD OUT", "FINALISTS"}


def headings(lines: list[str]) -> list[str]:
    """The section headings, in order. Not `str.isupper`: a table row whose
    only letters are its `E` numbers is upper case too."""
    return [line for line in lines if line in HEADINGS]


def section(lines: list[str], heading: str) -> list[str]:
    """The lines under one heading: past the blank line that follows it, up
    to the next blank line or heading. A table's header row and its rule are
    the first two."""
    start = lines.index(heading) + 2
    body = []
    for line in lines[start:]:
        if line == "" or line in HEADINGS:
            break
        body.append(line)
    return body


def examples_list(lines: list[str]) -> list[str]:
    """The numbered examples under the SCORES table: the block after the
    table's own blank line."""
    start = lines.index("SCORES") + 2
    after_table = lines.index("", start) + 1
    body = []
    for line in lines[after_table:]:
        if line == "":
            break
        body.append(line)
    return body


# ---------------------------------------------------------------- the shape


def test_three_sections_in_order_each_at_column_one():
    lines = front.render(DOCUMENT) + front.finalists(DOCUMENT)

    assert headings(lines) == ["SCORE", "SCORES", "FINALISTS"]
    assert not any(l.startswith(" ") for l in lines if l), "nothing indented"
    for heading in ("SCORE", "SCORES", "FINALISTS"):
        at = lines.index(heading)
        assert lines[at - 1] == "" and lines[at + 1] == "", "a blank line either side of a heading"


def test_the_summary_counts_candidates_examples_and_the_eligible():
    assert front.summary(DOCUMENT) == (
        "3 candidates over 3 validation examples, 2 best on at least one"
    )


def test_the_summary_says_one_rather_than_ones():
    document = {**DOCUMENT, "examples": ["only"], "pool": {**DOCUMENT["pool"], "candidates": 1}}
    assert front.summary(document) == (
        "1 candidate over 1 validation example, 2 best on at least one"
    )


# ---------------------------------------------------------------- the score


def test_the_score_section_is_the_one_sentence_wrapped():
    sentence = "word " * 40
    lines = front.render({**DOCUMENT, "score": sentence})
    body = section(lines, "SCORE")

    assert " ".join(body) == sentence.strip()
    assert len(body) > 1 and all(len(line) <= 78 for line in body), "wrapped"


def test_a_file_with_no_sentence_has_no_score_section():
    assert "SCORE" not in front.render({**DOCUMENT, "score": ""})


# --------------------------------------------------------------- the table


def test_the_table_is_candidates_against_examples_with_best_on_last():
    body = section(front.render(DOCUMENT), "SCORES")

    assert body[0].split() == ["cand", "val", "E1", "E2", "E3", "chars", "best", "on"]
    assert body[2].split() == ["seed", "0.667", "1.00", "1.00", "0.00", "2,076", "E1-E2"]
    assert body[3].split() == ["1", "0.333", "0.00", "1.00", "-", "1,900"]
    assert body[4].split() == ["2", "0.833", "0.50", "1.00", "1.00", "3,404", "E2-E3"]
    assert len(body) == 5, "one row per candidate in the pool, eligible or not"


def test_every_table_header_has_a_rule_under_it():
    header, rule = section(front.render(DOCUMENT), "SCORES")[:2]
    assert rule == "-" * len(header)


def test_numbers_line_up_under_their_headers():
    """Text left, numbers right within their columns, so the decimals stack.
    2,076 and 3,404 end under `chars`, and each score ends under its E."""
    body = section(front.render(DOCUMENT), "SCORES")
    header, _rule, seed, missing, other = body

    at = header.index("chars") + len("chars")
    assert seed[:at].endswith("2,076")
    assert other[:at].endswith("3,404")
    e3 = header.index("E3") + len("E3")
    assert seed[:e3].endswith("0.00")
    assert missing[:e3].endswith("-"), "a missing score is a dash in its own column"


def test_the_examples_are_numbered_under_the_table():
    assert examples_list(front.render(DOCUMENT)) == [
        "E1  first case",
        "E2  second case",
        "E3  third case",
    ]


def test_the_number_column_is_as_wide_as_the_widest_number():
    document = {
        **DOCUMENT,
        "examples": [f"case {n}" for n in range(1, 12)],
        "candidates": [
            {**DOCUMENT["candidates"][0], "scores": [1.0] * 11, "best_on": list(range(1, 12))}
        ],
    }
    listed = examples_list(front.render(document))

    assert listed[0] == "E1   case 1"
    assert listed[10] == "E11  case 11"


def test_best_on_is_written_as_ranges():
    assert front._ranges([1, 2, 3, 4, 5, 6, 7, 9, 10, 11]) == "E1-E7, E9-E11"
    assert front._ranges([8]) == "E8"
    assert front._ranges([3, 1, 2]) == "E1-E3", "the order they arrive in does not matter"
    assert front._ranges([]) == ""


def test_styled_highlights_the_top_cell_in_each_example_column():
    """Ties are all highlighted: on E2 every candidate scored 1.00, and each
    of them is eligible to be picked on that example."""
    import click

    styled = front.render(DOCUMENT, styled=True)
    at = [click.unstyle(line) for line in styled].index("SCORES")
    seed, nobody, longer = styled[at + 4:at + 7]
    top = click.style("1.00", fg="green", bold=True)

    assert seed.count(top) == 2, "E1 and E2"
    assert nobody.count(top) == 1, "E2 only; 0.00 on E1 is not the top"
    assert longer.count(top) == 2, "E2 and E3"
    assert click.style("0.50", fg="green", bold=True) not in longer


def test_an_overfit_file_adds_a_held_out_section_before_the_finalists():
    document = {
        **DOCUMENT,
        "holdout": {
            "cases": 34,
            "candidates": {
                "0": {"train": 0.821, "holdout": 0.997, "worse_than_seed_on": None},
                "2": {"train": 0.979, "holdout": 0.987, "worse_than_seed_on": 26},
            },
        },
    }

    lines = front.render(document) + front.finalists(document)
    body = section(lines, "HELD OUT")

    assert headings(lines) == ["SCORE", "SCORES", "HELD OUT", "FINALISTS"]
    assert body[0].split()[:3] == ["cand", "train", "holdout"]
    assert body[2] == "seed  0.821  0.997"
    assert body[3] == "2     0.979  0.987    26 of 34"


def test_a_file_without_a_held_out_block_has_no_held_out_section():
    assert "HELD OUT" not in front.render(DOCUMENT)


# ------------------------------------------------------------ the finalists


def test_finalists_are_the_eligible_candidates_previewed_seed_first():
    """Candidate 1 is best on no example, so GEPA would never pick it to
    mutate, and it is not a finalist."""
    body = front.finalists(DOCUMENT)

    assert body[:3] == ["", "FINALISTS", ""]
    assert body[3] == "seed (0.667)"
    assert body[4] == "the seed prompt"
    assert body[5] == ""
    assert body[6] == "candidate 2 (0.833)"
    assert body[7] == "the longer prompt second line", "newlines collapsed onto one line"
    assert "candidate 1 (0.333)" not in body


def test_a_long_prompt_is_cut_at_a_word_with_an_ellipsis():
    preview = front._preview("word " * 60)

    assert len(preview) <= front.PREVIEW + 2
    assert preview.endswith("word …")
    assert front._preview("short") == "short"


def only(components: dict[str, str]) -> dict:
    """The document with one eligible candidate holding these components."""
    return {
        **DOCUMENT,
        "candidates": [{**DOCUMENT["candidates"][2], "components": components}],
    }


def test_a_finalist_with_several_components_gets_a_line_per_component():
    body = front.finalists(only({"list_tables": "every table", "describe_table": "one table"}))

    assert body[4] == "[list_tables] every table"
    assert body[5] == "[describe_table] one table"


def test_whole_prints_the_text_as_written_with_its_line_breaks():
    """`make gepa-tools-pareto`: four descriptions fit on a screen, so they
    are read whole rather than cut at 120 characters, and `make
    gepa-config-pareto` is a YAML block, which collapsed onto one line is
    not YAML."""
    body = front.finalists(
        only({"config": "plan:\n  effort: low\nexplore:\n  effort: high"}), whole=True
    )

    assert body[4:8] == ["plan:", "  effort: low", "explore:", "  effort: high"]
    assert "…" not in "".join(body)


def test_whole_labels_each_component_on_its_own_line():
    body = front.finalists(
        only({"list_tables": "word " * 40, "describe_table": "one table"}), whole=True
    )

    assert body[4] == "[list_tables]"
    assert all(len(line) <= 78 for line in body), "wrapped, not cut"
    assert body[body.index("[describe_table]") + 1] == "one table"


def test_which_targets_read_whole():
    """Tool descriptions and the config block are read whole; a node prompt is
    pages, and shows how it opens."""
    assert front._whole_for("tools") is True
    assert front._whole_for("config") is True
    assert front._whole_for("extract") is False
    assert front._whole_for(None) is False


# ------------------------------------------------------------------ the files


def test_a_report_file_reads_back(tmp_path, capsys):
    """Where the file is and when it was written, the summary, then ABOUT
    for its target and every section in order."""
    path = tmp_path / "extract.pareto.json"
    path.write_text(json.dumps(DOCUMENT))

    assert front.main([str(path)]) == 0

    out = capsys.readouterr().out
    first, second = out.splitlines()[:2]
    assert first.startswith(f"{path}  (written 20")
    assert second == "3 candidates over 3 validation examples, 2 best on at least one"
    assert 'The "extract" node is responsible for' in out
    assert (
        out.index("ABOUT") < out.index("SCORE\n") < out.index("SCORES")
        < out.index("FINALISTS")
    )
    assert "\nseed  0.667" in out


def test_a_file_in_the_old_format_gets_one_line_saying_so(tmp_path, capsys):
    """The files committed before the report changed hold terms, not
    examples. There is no table to draw from them, and a KeyError traceback
    would not say what to do."""
    old = tmp_path / "tools.pareto.json"
    old.write_text(json.dumps({
        "target": "tools",
        "objectives": ["correct", "cost", "tool_calls"],
        "pool": {"candidates": 2, "on_front": 1},
        "front": [],
    }))

    assert front.main([str(old)]) == 1

    out = capsys.readouterr().out
    assert out == (
        f"{old}: written before the report changed to candidates against "
        f"validation examples — `make gepa-tools` writes a new one\n"
    )


def test_about_is_a_headed_wrapped_paragraph_or_nothing():
    lines = front.about("word " * 40)

    assert lines[:3] == ["", "ABOUT", ""]
    assert all(len(l) <= 78 for l in lines[3:]) and len(lines) > 4, "wrapped"
    assert front.about("") == []
    assert front._about_for("nope") == ""


def test_plain_when_not_a_terminal(tmp_path, capsys):
    """A pipe gets no colour codes. `styled` follows stdout, and pytest's
    capture is not a terminal."""
    path = tmp_path / "extract.pareto.json"
    path.write_text(json.dumps(DOCUMENT))
    front.main([str(path)])

    assert "\x1b[" not in capsys.readouterr().out


def test_styling_moves_no_column():
    import click

    plain = front.render(DOCUMENT) + front.finalists(DOCUMENT)
    styled = front.render(DOCUMENT, styled=True) + front.finalists(DOCUMENT, styled=True)

    assert [click.unstyle(l) for l in styled] == plain


def test_several_files_print_in_order_and_a_missing_one_does_not_stop_the_rest(
    tmp_path, capsys
):
    good = tmp_path / "a.pareto.json"
    good.write_text(json.dumps(DOCUMENT))
    junk = tmp_path / "b.pareto.json"
    junk.write_text(json.dumps({"no": "report"}))

    code = front.main([str(tmp_path / "missing.json"), str(junk), str(good)])

    out = capsys.readouterr().out
    assert code == 1
    assert "missing.json: no such file" in out
    assert "b.pareto.json: not a report" in out
    assert out.index("no such file") < out.index("not a report") < out.index("SCORES")


def test_no_files_says_so_rather_than_printing_nothing(capsys):
    """`make gepa-config-pareto` before any config run: the wildcard over
    tools/gepa/out/ is empty, and an empty command that exits 0 looks like a
    report with nothing in it."""
    assert front.main([]) == 1
    assert "no run of that target has left one" in capsys.readouterr().out


# -------------------------------------------------------------- the working row


WORKING = {"label": "config/prompts/extract.md", "val": 1.0, "scores": [1.0, 1.0, 1.0], "chars": 2500}


def test_the_working_text_is_one_more_row_under_the_pool_and_a_sentence():
    lines = front.render({**DOCUMENT, "working": WORKING})
    body = section(lines, "SCORES")

    assert body[-1].split() == ["working", "1.000", "1.00", "1.00", "1.00", "2,500"]
    assert len(body) == 6, "the pool's three rows and this one"
    at = lines.index(body[-1])
    assert lines[at + 1] == ""
    assert " ".join(lines[at + 2:at + 4]) == (
        "`working` is the text the agent runs, config/prompts/extract.md, scored "
        "on the same examples for comparison and not part of the search."
    )


def test_the_working_row_changes_nothing_else():
    """Not highlighted, best on nothing, and not a finalist. The summary and
    the pool's highlighting are what they were without it."""
    import click

    with_working = {**DOCUMENT, "working": WORKING}
    styled = front.render(with_working, styled=True)
    row = next(l for l in styled if click.unstyle(l).startswith("working"))

    assert row == click.unstyle(row), "never highlighted"
    assert front.summary(with_working) == front.summary(DOCUMENT)
    assert front.finalists(with_working) == front.finalists(DOCUMENT)
    assert examples_list(front.render(with_working)) != [], "the examples are still listed"


def test_a_document_without_the_working_key_renders_as_before():
    assert all("working" not in line for line in front.render(DOCUMENT))
    assert section(front.render(DOCUMENT), "SCORES")[0].startswith("cand  val")
