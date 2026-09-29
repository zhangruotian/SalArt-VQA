import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate
import providers
from metrics import (
    LABELS,
    QUESTIONS,
    normalize_answer,
    read_jsonl,
    read_predictions,
    score,
)


def test_full_dataset_denominators_and_perfect_answers():
    labels = read_jsonl(LABELS)
    predictions = {r["row_id"]: {q: r[f"{q}_answer"] for q in QUESTIONS} for r in labels}
    result = score(labels, predictions)
    assert result["questions"] == 3681
    assert result["missing_or_invalid_answers"] == 0
    assert [m["total"] for m in result["scores"].values()] == [475, 475, 475, 356, 356, 356, 119, 831, 831, 950]
    assert all(m["accuracy"] == 100 for m in result["scores"].values())
    assert all(m["correct"] == 0 for m in score(labels, {})["scores"].values())


def test_joint_metrics_require_all_answers_on_the_same_image():
    labels = read_jsonl(LABELS)
    predictions = {r["row_id"]: {q: r[f"{q}_answer"] for q in QUESTIONS} for r in labels}
    artifact = next(r for r in labels if r["image_role"] == "artifact")
    predictions[artifact["row_id"]]["q2"] = "not an answer"
    generated = next(r for r in labels if r["image_role"] == "paired_generated_counterpart")
    predictions[generated["row_id"]]["q1"] = "yes"  # Must not enter a denominator.
    table = score(labels, predictions)["scores"]
    assert table["artifact_q1"]["correct"] == 475
    assert table["artifact_all4"]["correct"] == 474
    assert table["overall_all4"]["correct"] == 830
    assert table["overall_all3"]["correct"] == 949
    assert table["real_all4"]["correct"] == 356


def test_score_command_exports_metrics_matching_the_leaderboard(tmp_path):
    labels = read_jsonl(LABELS)
    predictions = [{"row_id": r["row_id"], **{q: r[f"{q}_answer"] for q in QUESTIONS}}
                   for r in labels]
    artifact_index = next(i for i, r in enumerate(labels) if r["image_role"] == "artifact")
    predictions[artifact_index]["q2"] = None
    path = tmp_path / "predictions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in predictions))
    output = tmp_path / "scored"
    subprocess.run([sys.executable, "evaluate.py", "score", str(path), "--output", str(output)],
                   cwd=LABELS.parent.parent, capture_output=True, text=True, check=True)
    report = json.loads((output / "metrics.json").read_text())
    leaderboard = json.loads((LABELS.parent.parent / "leaderboard.json").read_text())
    assert len(report["scores"]) == 10
    assert all(set(report["scores"]) == set(model["scores"]) for model in leaderboard["models"])
    assert report["missing_or_invalid_answers"] == 1
    assert report["scores"]["overall_all4"]["correct"] == 830
    assert report["scores"]["overall_all4"]["total"] == 831
    assert report["scores"]["overall_all3"]["correct"] == 949
    assert report["scores"]["overall_all3"]["total"] == 950
    with (output / "metrics.csv").open() as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 1
    assert exported[0] == {key: f"{value['accuracy']:.2f}"
                           for key, value in report["scores"].items()}


def test_strict_answer_parsing_and_unknown_ids():
    assert normalize_answer(" YES\n", "q1") == "yes"
    assert normalize_answer(" b ", "q3") == "B"
    for value in (None, "A or B", '{"answer":"A"}', "I choose A.", "<think>A</think>B"):
        assert normalize_answer(value, "q2") is None
    with pytest.raises(ValueError, match="Unknown"):
        score(read_jsonl(LABELS), {"unknown": {}})


def test_hub_loading_fetches_only_the_needed_v1_shard(tmp_path, monkeypatch):
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq
    shard = tmp_path / "test.parquet"
    pq.write_table(pa.Table.from_pylist([{"row_id": "first"}, {"row_id": "second"}]), shard)
    calls = []
    def download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        return str(shard)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert list(evaluate.dataset_rows(None, 1)) == [{"row_id": "first"}]
    assert calls == [(evaluate.DATASET, "data/test-00000-of-00005.parquet",
                      {"repo_type": "dataset", "revision": evaluate.REVISION})]


def example_row(label):
    return {**label, "image": {"bytes": b"original", "path": "image.png"},
            "q3_overlay_image": {"bytes": b"overlay", "path": "overlay.png"},
            **{f"{q}_prompt": q if label[f"{q}_answer"] else None for q in QUESTIONS},
            "q2_options": dict(zip("ABCDE", ("one", "two", "three", "four", "none"))),
            "q4_options": dict(zip("ABCDE", ("a", "b", "c", "d", "e")))}


def test_question_inputs_exclude_labels_and_select_the_overlay():
    label = next(r for r in read_jsonl(LABELS) if r["image_role"] == "paired_generated_counterpart")
    calls = []
    def generate(prompt, image, mime):
        calls.append((prompt, image, mime))
        return {"text": "E"}
    client = SimpleNamespace(generate=generate, errors=(RuntimeError,))
    result = evaluate.evaluate_row(client, example_row(label), {})
    assert len(calls) == 3
    assert [c[1] for c in calls] == [b"original", b"overlay", b"original"]
    assert "Options:\nA. one" in calls[0][0]
    assert calls[1][0] == "q3"
    assert all(label["row_id"] not in c[0] and label["image_role"] not in c[0] for c in calls)
    assert "q1" not in result


def test_resume_preserves_completed_questions_and_retries_api_errors(tmp_path, monkeypatch):
    label = read_jsonl(LABELS)[0]
    row = example_row(label)
    calls = []
    fail = [True]
    class FakeClient:
        errors = (RuntimeError,)
        def __init__(self, *args):
            pass
        def close(self):
            pass
        def generate(self, prompt, image, mime):
            q = prompt[:2]
            calls.append(q)
            if q == "q2" and fail[0]:
                raise RuntimeError("Temporary service error")
            return {"text": label[f"{q}_answer"]}
    monkeypatch.setattr(providers, "Client", FakeClient)
    monkeypatch.setattr(evaluate, "dataset_rows", lambda *_: iter([deepcopy(row)]))
    args = SimpleNamespace(provider="openai", model="test", base_url=None, max_tokens=4096,
                           temperature=None, request_options={}, limit=1, output=tmp_path,
                           resume=False, workers=1, data_dir=None)
    with pytest.raises(SystemExit, match="API error"):
        evaluate.run(args)
    assert calls == ["q1", "q2"]
    assert not json.loads((tmp_path / "metrics.json").read_text())["complete"]
    fail[0] = False
    args.resume = True
    evaluate.run(args)
    assert calls == ["q1", "q2", "q2", "q3", "q4"]
    report = json.loads((tmp_path / "metrics.json").read_text())
    assert report["complete"] and report["scope"] == "subset"
    assert report["scores"]["artifact_all4"]["accuracy"] == 100
    assert (tmp_path / "metrics.csv").is_file()
    assert len(read_predictions(tmp_path / "predictions.jsonl")) == 1
    evaluate.run(args)
    assert len(calls) == 5  # Fully completed resume makes zero calls.
    args.model = "different"
    with pytest.raises(ValueError, match="same model"):
        evaluate.run(args)
