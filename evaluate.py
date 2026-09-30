"""Evaluate your model on SalArt-VQA v1 and compute benchmark metrics."""

import argparse
import json
import mimetypes
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path

from metrics import (
    LABELS,
    QUESTIONS,
    normalize_answer,
    read_jsonl,
    read_predictions,
    score,
    show_report,
    write_report,
)

DATASET = "salartvqa/SalArt-VQA"
# Immutable commit behind the v1 release tag.
REVISION = "eacc6d39661b04c0ac2abdcd8ed2c5d37d9ed6f4"


def build_prompt(row, question):
    prompt = row[f"{question}_prompt"]
    if question in ("q2", "q4"):
        options = row[f"{question}_options"]
        prompt += "\n\nOptions:\n" + "\n".join(f"{key}. {options[key]}" for key in "ABCDE")
    return prompt


def dataset_rows(data_dir, limit):
    import pyarrow.parquet as pq
    if data_dir is None:
        from huggingface_hub import hf_hub_download
        files = (hf_hub_download(DATASET, f"data/test-{i:05d}-of-00005.parquet",
                                 repo_type="dataset", revision=REVISION) for i in range(5))
    else:
        files = sorted((data_dir / "data").glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No data/*.parquet in {data_dir}")
    rows = (row for file in files for batch in pq.ParquetFile(file).iter_batches(batch_size=1)
            for row in batch.to_pylist())
    yield from islice(rows, limit)


def evaluate_row(client, row, previous):
    result = {"row_id": row["row_id"], "raw": {}, **previous}
    for question in QUESTIONS:
        if row[f"{question}_answer"] is None:
            continue
        if question in result["raw"] and "error" not in result["raw"][question]:
            continue
        image = row["q3_overlay_image" if question == "q3" else "image"]
        mime_type = mimetypes.guess_type(image["path"])[0]
        try:
            response = client.generate(build_prompt(row, question), image["bytes"], mime_type)
        except client.errors as error:
            response = {"text": "", "error": f"{type(error).__name__}: {str(error)[:500]}"}
        result["raw"][question] = response
        result[question] = normalize_answer(response["text"], question)
        if "error" in response:
            break
    return result


def run(args):
    from providers import Client
    labels = read_jsonl(LABELS)[:args.limit]
    config = {"dataset": DATASET, "revision": REVISION, "provider": args.provider,
              "model": args.model, "base_url": args.base_url, "max_tokens": args.max_tokens,
              "temperature": args.temperature, "request_options": args.request_options,
              "limit": args.limit}
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "run.json"
    path = args.output / "predictions.jsonl"
    if args.resume:
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Resume requires the same model, dataset and request settings.")
        predictions = read_predictions(path) if path.exists() else {}
    else:
        if config_path.exists() or path.exists():
            raise ValueError("Output already contains a run. Use --resume or a new --output.")
        predictions = {}
    client = Client(args.provider, args.model, args.base_url, args.max_tokens,
                    args.temperature, args.request_options)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    expected = {row["row_id"]: row for row in labels}
    completed = {key for key, result in predictions.items()
                 if all(q in result.get("raw", {}) and "error" not in result["raw"][q]
                        for q in QUESTIONS if expected[key][f"{q}_answer"] is not None)}
    rows = (row for row in dataset_rows(args.data_dir, args.limit) if row["row_id"] not in completed)
    failed = False
    seen = set(completed)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool, path.open("a") as stream:
            while batch := list(islice(rows, args.workers)):
                for row in batch:
                    if row["row_id"] not in expected or row["row_id"] in seen:
                        raise ValueError("Dataset does not match the v1 labels.")
                    seen.add(row["row_id"])
                    for q in QUESTIONS:
                        if row[f"{q}_answer"] != expected[row["row_id"]][f"{q}_answer"]:
                            raise ValueError("Dataset labels do not match v1.")
                pending = [pool.submit(evaluate_row, client, row, predictions.get(row["row_id"], {}))
                           for row in batch]
                for future in as_completed(pending):
                    result = future.result()
                    predictions[result["row_id"]] = result
                    stream.write(json.dumps(result) + "\n")
                    stream.flush()
                    errors = [r["error"] for r in result["raw"].values() if "error" in r]
                    failed |= bool(errors)
                    for error in errors:
                        print(error, file=sys.stderr)
                    print(f"Saved {len(predictions)}/{len(labels)} images", flush=True)
                if failed:
                    break
    finally:
        client.close()
        report = score(labels, predictions)
        report["scope"] = "full" if args.limit is None else "subset"
        report["complete"] = all(
            q in predictions.get(key, {}).get("raw", {})
            and "error" not in predictions[key]["raw"][q]
            for key, row in expected.items() for q in QUESTIONS
            if row[f"{q}_answer"] is not None
        )
        write_report(report, args.output)
        show_report(report)
    if failed:
        raise SystemExit("API error: saved partial results. Fix the issue and rerun with --resume.")
    if seen != set(expected):
        raise ValueError("Dataset is incomplete; metrics include missing answers as incorrect.")


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be at least 1")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scorer = sub.add_parser("score", help="Score JSONL predictions against all 950 v1 images")
    scorer.add_argument("predictions", type=Path)
    scorer.add_argument("--output", type=Path, default=Path("runs/scored"))
    runner = sub.add_parser("run", help="Evaluate a model on v1 (3,681 independent calls for a full run)")
    from providers import PROVIDERS
    runner.add_argument("--provider", choices=PROVIDERS, required=True)
    runner.add_argument("--model", required=True)
    runner.add_argument("--base-url")
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--workers", type=positive_int, default=4)
    runner.add_argument("--max-tokens", type=positive_int, default=4096)
    runner.add_argument("--temperature", type=float, help="Omit for the provider default")
    runner.add_argument("--request-options", type=json.loads, default={},
                        help="JSON: SDK extra_body, or Gemini GenerateContentConfig fields")
    runner.add_argument("--limit", type=positive_int, help="First N images, for a small smoke test")
    runner.add_argument("--data-dir", type=Path, help="Local v1 snapshot containing data/*.parquet")
    runner.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "score":
        report = score(read_jsonl(LABELS), read_predictions(args.predictions))
        write_report(report, args.output)
        show_report(report)
    else:
        run(args)


if __name__ == "__main__":
    main()
