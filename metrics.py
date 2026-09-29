"""SalArt-VQA benchmark metrics, matching the leaderboard."""

import csv
import json
from pathlib import Path

QUESTIONS = ("q1", "q2", "q3", "q4")
ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "data" / "labels.jsonl"
# Key, image role (None means all), questions that must ALL be correct.
METRICS = (
    ("artifact_q1", "artifact", ("q1",)),
    ("artifact_all4", "artifact", QUESTIONS),
    ("artifact_all3", "artifact", QUESTIONS[1:]),
    ("real_q1", "clean_reference", ("q1",)),
    ("real_all4", "clean_reference", QUESTIONS),
    ("real_all3", "clean_reference", QUESTIONS[1:]),
    ("generated_all3", "paired_generated_counterpart", QUESTIONS[1:]),
    ("overall_q1", None, ("q1",)),
    ("overall_all4", None, QUESTIONS),
    ("overall_all3", None, QUESTIONS[1:]),
)


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_predictions(path):
    # Resumed runs append updated rows; the last record for an ID is authoritative.
    return {row["row_id"]: row for row in read_jsonl(path)}


def normalize_answer(text, question):
    if not isinstance(text, str):
        return None
    value = text.strip().lower() if question == "q1" else text.strip().upper()
    allowed = ("yes", "no") if question == "q1" else tuple("ABCDE")
    return value if value in allowed else None


def score(labels, predictions):
    ids = {row["row_id"] for row in labels}
    unknown = set(predictions) - ids
    if unknown:
        raise ValueError(f"Unknown prediction IDs: {sorted(unknown)[:3]}")
    correct = {}
    invalid = 0
    questions = 0
    for row in labels:
        answers = {}
        for question in QUESTIONS:
            gold = row[f"{question}_answer"]
            if gold is None:
                continue
            prediction = normalize_answer(predictions.get(row["row_id"], {}).get(question), question)
            invalid += prediction is None
            questions += 1
            answers[question] = prediction == gold
        correct[row["row_id"]] = answers

    table = {}
    for key, role, required in METRICS:
        eligible = [row for row in labels
                    if (role is None or row["image_role"] == role)
                    and all(q in correct[row["row_id"]] for q in required)]
        count = sum(all(correct[row["row_id"]][q] for q in required) for row in eligible)
        table[key] = {"correct": count, "total": len(eligible),
                      "accuracy": 100 * count / len(eligible) if eligible else None}
    return {"rows": len(labels), "questions": questions,
            "prediction_rows": len(predictions),
            "missing_or_invalid_answers": invalid, "scores": table}


def write_report(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    with (directory / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(report["scores"])
        writer.writerow("" if m["accuracy"] is None else f'{m["accuracy"]:.2f}'
                        for m in report["scores"].values())


def show_report(report):
    print(f"{report['rows']} images, {report['questions']} questions; "
          f"{report['missing_or_invalid_answers']} missing/invalid answers")
    for key, metric in report["scores"].items():
        value = "n/a" if metric["accuracy"] is None else f"{metric['accuracy']:.2f}%"
        print(f"{key:20s} {value:>8s}  ({metric['correct']}/{metric['total']})")
