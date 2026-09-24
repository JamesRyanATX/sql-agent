"""A committed front read back for a person.

The search prints its front to stderr once and writes it to a file. The file
is what the talk points at, so the display has to come back off the file:
a legend, the front, how each finalist opens, and the held-out table for an
overfit run's file. Three headed sections, every one starting at column one,
so there is a left edge for the eye.

Pure: a dict in, lines out, plus tests over the real committed file.
"""

from __future__ import annotations

import json

from tools.gepa import front

DOCUMENT = {
    "target": "extract",
    "objectives": ["grounding", "census", "cost"],
    "weights": {"grounding": 0.5, "census": 0.3, "cost": 0.2},
    "pool": {"candidates": 3, "on_front": 2, "metric_calls": 40, "seed_index": 0},
    "best_per_objective": {"grounding": 1.0, "census": 1.0, "cost": 0.99},
    "front": [
        {
            "index": 0, "seed": True, "val": 0.958,
            "scores": {"grounding": 0.91, "census": 1.0, "cost": 0.99},
            "tops": ["census", "cost"], "chars": 2076,
            "components": {"extract": "the seed prompt"},
        },
        {
            "index": 2, "seed": False, "val": 0.966,
            "scores": {"grounding": 1.0, "census": 1.0, "cost": 0.94},
            "tops": ["grounding", "census"], "chars": 3404,
            "components": {"extract": "the longer prompt\nsecond line"},
        },
    ],
}
WORDS = {"grounding": "the SQL really ran", "census": "no counts", "cost": "cheaper"}


def section(lines: list[str], heading: str) -> list[str]:
    """The lines under one heading: past the blank line that follows it, up
    to the next blank line or heading. The header row and its rule are the
    first two."""
    start = lines.index(heading) + 2
    body = []
    for line in lines[start:]:
        if line == "" or line.isupper():
            break
        body.append(line)
    return body


# ---------------------------------------------------------------- the shape


def test_three_sections_in_order_each_at_column_one():
    lines = front.render(DOCUMENT, legend=WORDS) + front.finalists(DOCUMENT)

    assert [l for l in lines if l.isupper()] == ["LEGEND", "SCORES", "FINALISTS"]
    assert not any(l.startswith(" ") for l in lines if l), "nothing indented"
    for heading in ("LEGEND", "SCORES", "FINALISTS"):
        at = lines.index(heading)
        assert lines[at - 1] == "" and lines[at + 1] == "", "a blank line either side of a heading"


def test_every_table_header_has_a_rule_under_it():
    lines = front.render(DOCUMENT, legend=WORDS)

    for heading in ("LEGEND", "SCORES"):
        header, rule = section(lines, heading)[:2]
        assert rule == "-" * len(header)


def test_the_legend_leads_with_the_term_then_its_weight_then_its_words():
    body = section(front.render(DOCUMENT, legend=WORDS), "LEGEND")

    assert body[0].split() == ["term", "weight", "what", "a", "perfect", "score", "means"]
    assert body[2] == "grounding    0.50  the SQL really ran"
    assert body[3] == "census       0.30  no counts"
    assert body[4] == "cost         0.20  cheaper"
    assert len(body) == 5, "the terms and nothing under them"


def test_a_term_without_words_shows_its_weight_alone():
    body = section(front.render(DOCUMENT, legend={"census": "no counts"}), "LEGEND")

    assert body[2] == "grounding    0.50"
    assert body[3] == "census       0.30  no counts"


def test_no_legend_and_no_weights_prints_no_legend():
    document = {**DOCUMENT, "weights": {}}
    assert "LEGEND" not in front.render(document)


def test_the_front_names_the_seed_and_puts_best_on_in_a_column():
    body = section(front.render(DOCUMENT), "SCORES")

    assert body[0].split() == ["cand", "val", "grounding", "census", "cost", "chars", "best", "on"]
    assert body[2].split() == ["seed", "0.958", "0.91", "1.00", "0.99", "2,076", "census,", "cost"]
    assert body[3].split() == ["2", "0.966", "1.00", "1.00", "0.94", "3,404", "grounding,", "census"]
    assert len(body) == 4, "no `best` row: the cells say who wins each column"
    assert body[2].startswith("seed") and body[3].startswith("2 ")


def test_numbers_line_up_under_their_headers():
    """Text left, numbers right within their columns, so the decimals stack.
    2,076 and 3,404 end under `chars`."""
    body = section(front.render(DOCUMENT), "SCORES")
    header, _rule, seed, other = body[:4]

    at = header.index("chars") + len("chars")
    assert seed[:at].endswith("2,076")
    assert other[:at].endswith("3,404")


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

    assert [l for l in lines if l.isupper()] == ["LEGEND", "SCORES", "HELD OUT", "FINALISTS"]
    assert body[0].split()[:3] == ["cand", "train", "holdout"]
    assert body[2] == "seed  0.821  0.997"
    assert body[3] == "2     0.979  0.987    26 of 34"


def test_a_file_without_a_held_out_block_has_no_held_out_section():
    assert "HELD OUT" not in front.render(DOCUMENT)


# ------------------------------------------------------------ the finalists


def test_finalists_are_previewed_seed_first_with_the_text_under_the_name():
    body = front.finalists(DOCUMENT)

    assert body[:3] == ["", "FINALISTS", ""]
    assert body[3] == "seed (0.958)"
    assert body[4] == "the seed prompt"
    assert body[5] == ""
    assert body[6] == "candidate 2 (0.966)"
    assert body[7] == "the longer prompt second line", "newlines collapsed onto one line"


def test_a_long_prompt_is_cut_at_a_word_with_an_ellipsis():
    preview = front._preview("word " * 60)

    assert len(preview) <= front.PREVIEW + 2
    assert preview.endswith("word …")
    assert front._preview("short") == "short"


def test_a_finalist_with_several_components_gets_a_line_per_component():
    document = {
        **DOCUMENT,
        "front": [
            {
                **DOCUMENT["front"][1],
                "components": {"list_tables": "every table", "describe_table": "one table"},
            }
        ],
    }

    body = front.finalists(document)

    assert body[4] == "[list_tables] every table"
    assert body[5] == "[describe_table] one table"


def test_whole_prints_the_text_as_written_with_its_line_breaks():
    """`make gepa-tools-pareto`: four descriptions fit on a screen, so they
    are read whole rather than cut at 120 characters, and `make
    gepa-config-pareto` is a YAML block, which collapsed onto one line is
    not YAML."""
    document = {
        **DOCUMENT,
        "front": [
            {
                **DOCUMENT["front"][1],
                "components": {
                    "config": "plan:\n  effort: low\nexplore:\n  effort: high",
                },
            }
        ],
    }

    body = front.finalists(document, whole=True)

    assert body[4:8] == ["plan:", "  effort: low", "explore:", "  effort: high"]
    assert "…" not in "".join(body)


def test_whole_labels_each_component_on_its_own_line():
    document = {
        **DOCUMENT,
        "front": [
            {
                **DOCUMENT["front"][1],
                "components": {"list_tables": "word " * 40, "describe_table": "one table"},
            }
        ],
    }

    body = front.finalists(document, whole=True)

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


def test_the_committed_extract_front_reads_back(capsys):
    """The file section 4 names: its six terms with the metric's words, the
    seed row, and the seed's opening from the file's own copy."""
    assert front.main(["demo/gepa/extract.pareto.json"]) == 0

    out = capsys.readouterr().out
    first, second = out.splitlines()[:2]
    assert first.startswith("demo/gepa/extract.pareto.json  (written 20")
    assert second == "6 objectives over 3 candidates, 2 on the front"
    assert "grounding    0.35  every note that quotes SQL" in out
    assert "length       0.10  the prompt is no longer than" in out
    assert out.index("ABOUT") < out.index("LEGEND") < out.index("SCORES") < out.index("FINALISTS")
    assert 'The "extract" node is responsible for' in out
    assert "\nseed  0.966" in out

    from app import prompts

    opening = " ".join(prompts.get("extract").split())[:60]
    assert opening in out
    assert prompts.get("extract").strip() not in out, "a preview, not the whole thing"


def test_about_is_a_headed_wrapped_paragraph_or_nothing():
    lines = front.about("word " * 40)

    assert lines[:3] == ["", "ABOUT", ""]
    assert all(len(l) <= 78 for l in lines[3:]) and len(lines) > 4, "wrapped"
    assert front.about("") == []
    assert front._about_for("nope") == ""


def test_the_words_come_from_the_files_target():
    from tools.gepa import metric_extract, metric_turn

    assert front._words_for("extract") == metric_extract.LEGEND
    assert front._words_for("tools") == metric_turn.LEGEND
    assert front._words_for("config") == metric_turn.LEGEND
    assert front._words_for("nope") == {}


def test_plain_when_not_a_terminal(capsys):
    """A pipe gets no colour codes. `styled` follows stdout, and pytest's
    capture is not a terminal."""
    front.main(["demo/gepa/extract.pareto.json"])

    assert "\x1b[" not in capsys.readouterr().out


def test_styling_moves_no_column():
    import click

    plain = front.render(DOCUMENT, legend=WORDS) + front.finalists(DOCUMENT)
    styled = front.render(DOCUMENT, legend=WORDS, styled=True) + front.finalists(DOCUMENT, styled=True)

    assert [click.unstyle(l) for l in styled] == plain


def test_several_files_print_in_order_and_a_missing_one_does_not_stop_the_rest(
    tmp_path, capsys
):
    good = tmp_path / "a.pareto.json"
    good.write_text(json.dumps(DOCUMENT))
    junk = tmp_path / "b.pareto.json"
    junk.write_text(json.dumps({"no": "front"}))

    code = front.main([str(tmp_path / "missing.json"), str(junk), str(good)])

    out = capsys.readouterr().out
    assert code == 1
    assert "missing.json: no such file" in out
    assert "b.pareto.json: not a front" in out
    assert out.index("no such file") < out.index("not a front") < out.index("SCORES")


def test_no_files_says_so_rather_than_printing_nothing(capsys):
    """`make gepa-config-pareto` before any config run: the wildcard over
    tools/gepa/out/ is empty, and an empty command that exits 0 looks like a
    front with nothing on it."""
    assert front.main([]) == 1
    assert "no run of that target has left one" in capsys.readouterr().out
