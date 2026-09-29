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


def test_paper_recomputed_from_archived_predictions():
    result = subprocess.run([sys.executable, "evaluate.py", "paper"], cwd=evaluate.ROOT,
                            capture_output=True, text=True, check=True)
    assert len(result.stdout.splitlines()) == 11
    assert "Gemini 3.1 Flash Lite overall_all3: paper 24.73, recomputed 24.74" in result.stderr
    assert "Gemma-4-31B-it overall_all3: paper 20.31, recomputed 20.32" in result.stderr


def test_full_dataset_denominators_and_perfect_answers():
    labels = read_jsonl(LABELS)
    predictions = {r["row_id"]: {q: r[f"{q}_answer"] for q in QUESTIONS} for r in labels}
    result = score(labels, predictions)
    assert result["questions"] == 3681
    assert result["missing_or_invalid_answers"] == 0
    assert [m["total"] for m in result["table1"].values()] == [475, 475, 475, 356, 356, 356, 119, 831, 831, 950]
    assert all(m["accuracy"] == 100 for m in result["table1"].values())
    assert all(m["correct"] == 0 for m in score(labels, {})["table1"].values())


def test_joint_metrics_require_all_answers_on_the_same_image():
    labels = read_jsonl(LABELS)
    predictions = {r["row_id"]: {q: r[f"{q}_answer"] for q in QUESTIONS} for r in labels}
    artifact = next(r for r in labels if r["image_role"] == "artifact")
    predictions[artifact["row_id"]]["q2"] = "not an answer"
    generated = next(r for r in labels if r["image_role"] == "paired_generated_counterpart")
    predictions[generated["row_id"]]["q1"] = "yes"  # Must not enter a denominator.
    table = score(labels, predictions)["table1"]
    assert table["artifact_q1"]["correct"] == 475
    assert table["artifact_all4"]["correct"] == 474
    assert table["overall_all4"]["correct"] == 830
    assert table["overall_all3"]["correct"] == 949
    assert table["real_all4"]["correct"] == 356


def test_strict_answer_parsing_and_unknown_ids():
    assert normalize_answer(" YES\n", "q1") == "yes"
    assert normalize_answer(" b ", "q3") == "B"
    for value in (None, "A or B", '{"answer":"A"}', "I choose A.", "<think>A</think>B"):
        assert normalize_answer(value, "q2") is None
    with pytest.raises(ValueError, match="Unknown"):
        score(read_jsonl(LABELS), {"unknown": {}})


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
    assert report["table1"]["artifact_all4"]["accuracy"] == 100
    assert len(read_predictions(tmp_path / "predictions.jsonl")) == 1
    evaluate.run(args)
    assert len(calls) == 5  # Fully completed resume makes zero calls.
    args.model = "different"
    with pytest.raises(ValueError, match="same model"):
        evaluate.run(args)
