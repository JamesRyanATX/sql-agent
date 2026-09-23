"""A committed front read back as the table the search printed.

The search prints its front to stderr once and writes it to a file. The file
is what the talk points at, so the table has to come back off the file the
same shape — and the held-out table too, for an overfit run's file.

Pure: a dict in, lines out, plus one test over the real committed file.
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
            "tops": ["census", "cost"], "chars": 2076, "components": {},
        },
        {
            "index": 2, "seed": False, "val": 0.966,
            "scores": {"grounding": 1.0, "census": 1.0, "cost": 0.94},
            "tops": ["grounding", "census"], "chars": 3404, "components": {},
        },
    ],
}


def test_the_table_has_the_shape_the_search_printed():
    lines = front.render(DOCUMENT)

    assert lines[0].strip() == "pareto    3 objectives over 3 candidates, 2 on the front"
    assert lines[1].split() == ["cand", "val", "grounding", "census", "cost", "chars"]
    assert lines[2].split() == ["0", "0.958", "0.91", "1.00", "0.99", "2,076", "seed", "tops", "census,", "cost"]
    assert lines[3].split()[:6] == ["2", "0.966", "1.00", "1.00", "0.94", "3,404"]
    assert "seed" not in lines[3]
    assert lines[4].split() == ["best", "-", "1.00", "1.00", "0.99"]


def test_the_chars_column_is_the_one_no_term_could_see():
    """Section 4's point is a column the metric did not have. It is on every
    row, right-aligned under its header, so 2,076 and 3,404 line up."""
    lines = front.render(DOCUMENT)
    header, seed, other = lines[1], lines[2], lines[3]

    at = header.index("chars") + len("chars")
    assert seed[:at].endswith("2,076")
    assert other[:at].endswith("3,404")


def test_an_overfit_file_prints_its_held_out_table_too():
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

    lines = front.render(document)
    text = "\n".join(lines)

    assert "holdout   34 cases the search never saw" in text
    assert "worse than the seed on" in text
    assert "   0   0.821    0.997  seed" in text
    assert "   2   0.979    0.987  26 of 34" in text


def test_a_file_without_a_held_out_block_prints_one_table():
    assert "holdout" not in "\n".join(front.render(DOCUMENT))


# ------------------------------------------------------------------ the files


def test_the_committed_extract_front_reads_back(capsys):
    """The file section 4 names. Its target, its six terms, and the seed row."""
    assert front.main(["demo/gepa/extract.pareto.json"]) == 0

    out = capsys.readouterr().out
    first, second = out.splitlines()[:2]
    assert first.startswith("demo/gepa/extract.pareto.json  (written 20")
    assert second.startswith("extract: grounding 0.35")
    assert "grounding  census   names   shape    cost  length   chars" in out
    assert "seed" in out


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
    assert out.index("no such file") < out.index("not a front") < out.index("pareto    3 objectives")


def test_no_files_says_so_rather_than_printing_nothing(capsys):
    """`make gepa-config-pareto` before any config run: the wildcard over
    tools/gepa/out/ is empty, and an empty command that exits 0 looks like a
    front with nothing on it."""
    assert front.main([]) == 1
    assert "no run of that target has left one" in capsys.readouterr().out


# ------------------------------------------------------------ the candidates
#
# The table says a candidate is 64% longer and bought grounding with it. The
# text is where that is read, so the display prints every front candidate in
# full, seed first.


def test_every_front_candidate_is_printed_in_full_seed_first():
    document = {
        **DOCUMENT,
        "front": [
            {**DOCUMENT["front"][1], "components": {"extract": "the longer prompt\nsecond line"}},
            {**DOCUMENT["front"][0], "components": {"extract": "the seed prompt"}},
        ],
    }

    text = "\n".join(front.candidates(document))

    assert text.index("seed  val 0.958  2,076 chars") < text.index("candidate 2  val 0.966  3,404 chars")
    assert "the seed prompt" in text
    assert "the longer prompt\nsecond line" in text


def test_a_candidate_with_several_components_gets_a_heading_per_component():
    document = {
        **DOCUMENT,
        "front": [
            {
                **DOCUMENT["front"][1],
                "components": {"list_tables": "every table", "describe_table": "one table"},
            }
        ],
    }

    text = "\n".join(front.candidates(document))

    assert "[list_tables]" in text and "[describe_table]" in text
    assert text.index("[list_tables]") < text.index("every table") < text.index("[describe_table]")


def test_the_display_prints_the_prompts_after_the_table(capsys):
    assert front.main(["demo/gepa/extract.pareto.json"]) == 0

    out = capsys.readouterr().out
    assert out.index("best") < out.index("── seed") < out.index("── candidate 2")
    # The seed's prompt is on disk too; the file's copy is what ran.
    from app import prompts

    assert prompts.get("extract").strip() in out


def test_plain_when_not_a_terminal(capsys):
    """A pipe gets no colour codes. `styled` follows stdout, and pytest's
    capture is not a terminal."""
    front.main(["demo/gepa/extract.pareto.json"])

    assert "\x1b[" not in capsys.readouterr().out
