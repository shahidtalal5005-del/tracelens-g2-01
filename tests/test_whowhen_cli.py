import json
import os

import pytest

from tracelens import cli
from tracelens.eval import whowhen
from tracelens.llm import FakeCompleter


def _rows(base=0, n=6):
    rows = []
    for i in range(n):
        hist = [{"name": "Orchestrator" if k % 2 == 0 else "WebSurfer", "content": f"message {k} " + ("Traceback error" if k == 3 else "ok")}
                for k in range(8)]
        step = 3 + base
        rows.append({"question": f"q{i}", "history": hist, "mistake_agent": hist[3]["name"], "mistake_step": str(step),
                     "mistake_reason": "r", "ground_truth": "x"})
    return rows


@pytest.mark.parametrize("base", [0, 1])
def test_index_base_autodetect(base):
    rows = _rows(base)
    assert whowhen.detect_index_base(rows) == base
    traces = whowhen.to_traces(rows)
    assert all(t.labels["fault"]["critical_event"] == 3 for t in traces)


def test_load_json_and_evaluate(tmp_path):
    p = tmp_path / "Hand-Crafted.json"
    p.write_text(json.dumps(_rows(0)))
    assert "usable=6" in whowhen.inspect(str(p))
    res = whowhen.evaluate(str(p), str(tmp_path / "out"))
    by = {r["method"]: r for r in res["rows"]}
    assert by["first_error_signal"]["step_acc"] == 1.0          # message 3 holds the keyword
    assert os.path.exists(tmp_path / "out" / "whowhen_results.md")


def test_history_given_as_json_string_and_judge(tmp_path):
    rows = _rows(0, 3)
    for r in rows:
        r["history"] = json.dumps(r["history"])
    p = tmp_path / "x.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    res = whowhen.evaluate(str(p), str(tmp_path / "o"), FakeCompleter(lambda s: '{"step": 3}'))
    by = {r["method"]: r for r in res["rows"]}
    assert by["llm_all_at_once"]["step_acc"] == 1.0


def test_missing_data_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        whowhen.load_rows(str(tmp_path))


def test_cli_end_to_end(tmp_path, capsys):
    out = str(tmp_path / "res")
    cli.main(["all", "--out", out, "--tasks-per-type", "3"])
    for f in ("bench.jsonl", "phase1_results.md", "phase1_results.json", "phase2_results.md", "phase2_results.json", "viewer.html"):
        assert os.path.getsize(os.path.join(out, f)) > 100, f
    html = open(os.path.join(out, "viewer.html"), encoding="utf-8").read()
    assert "critical-trace summary" in html.lower()
    r = json.load(open(os.path.join(out, "phase2_results.json")))
    assert {"gap1_attribution", "gap1_ablation", "gap2_observability", "gap3_summaries"} <= set(r)


def test_cli_run_agent_scripted(tmp_path, capsys):
    cli.main(["run-agent", "--n", "4", "--fault", "calc_error", "--out", str(tmp_path)])
    assert "runs" in capsys.readouterr().out
