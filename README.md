# SalArt-VQA

**Diagnosing Whether VLMs Understand Salient Artifacts in Generated Images** · NeurIPS 2026

[Paper](https://arxiv.org/abs/2606.12671) · [Dataset](https://huggingface.co/datasets/salartvqa/SalArt-VQA) · [Leaderboard](https://zhangruotian.github.io/SalArt-VQA/)

950 images and 3,681 independent visual questions: artifact detection (Q1), region selection (Q2), box grounding (Q3), and evidence selection (Q4).

## Recompute Table 1

The ten featured models' archived answers and v1 labels are included. No installation, downloads, API keys, or GPU are needed for scoring (Python 3.10+):

```bash
git clone https://github.com/zhangruotian/SalArt-VQA.git
cd SalArt-VQA
python evaluate.py paper
```

This computes the metrics from individual answers. The [interactive leaderboard](https://zhangruotian.github.io/SalArt-VQA/) shows published values; click any column to sort, search a model, or filter by access category.

Two published Overall All Q2–4 cells differ by 0.01 percentage points from the archived answers: Gemini 3.1 Flash Lite is **235/950 = 24.74%** (paper: 24.73%); Gemma-4-31B-it is **193/950 = 20.32%** (paper: 20.31%). The scorer reports the recomputed values.

## Evaluate a model

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY='your-key'
python evaluate.py run --provider openai --model gpt-5.4-nano \
  --output runs/gpt-5.4-nano --limit 1
```

Images download automatically from the self-contained HF **v1** snapshot as needed (~1.13 GB for the full dataset). `--limit 1` tests one image; **remove it for the full 3,681-question evaluation**. Use `--workers 4` to control concurrent images. Each question is a separate request with no shared conversation history.

Choose a provider and model below; keep the same `run` command and set a separate `--output` directory:

| Service | Arguments | Key environment variable |
|---|---|---|
| OpenAI Responses | `--provider openai --model gpt-5.4` | `OPENAI_API_KEY` |
| Claude Messages | `--provider anthropic --model claude-opus-4-6` | `ANTHROPIC_API_KEY` |
| Gemini | `--provider gemini --model gemini-3.1-pro-preview --temperature 0` | `GEMINI_API_KEY` |
| Kimi | `--provider kimi --model kimi-k2.5` | `MOONSHOT_API_KEY` |
| Ollama | `--provider ollama --model qwen3-vl:8b --temperature 0` | None locally |
| vLLM | `--provider vllm --model Qwen/Qwen3-VL-8B-Instruct --temperature 0` | `VLLM_API_KEY` if enabled |
| Other OpenAI-compatible servers | `--provider openai-compatible --model MODEL --base-url URL` | `OPENAI_API_KEY` if enabled |

Start a local vision model before using Ollama or vLLM:

```bash
ollama pull qwen3-vl:8b  # Ollama must be running; default endpoint: localhost:11434/v1
# Or, in a separate environment with vLLM installed and a suitable GPU:
vllm serve Qwen/Qwen3-VL-8B-Instruct --max-model-len 16384
```

`--base-url` overrides the endpoint (vLLM defaults to `http://localhost:8000/v1`; Kimi to `https://api.moonshot.ai/v1`). Model availability depends on the service. Select a vision-capable model.

The paper uses temperature 0 where supported and the provider default otherwise. Pass `--temperature 0` when supported; omitting it preserves the provider default. Reasoning settings are left unchanged. The output limit defaults to 4,096 tokens; use `--max-tokens` if needed. Provider-specific parameters go in `--request-options`, e.g. `'{"chat_template_kwargs":{"enable_thinking":false}}'` for a vLLM model supporting that setting. These are SDK `extra_body` fields, or Gemini `GenerateContentConfig` fields.

New runs use the frozen **v1 prompts and supplied Q3 overlays**. Historical paper runs used a JSON explanation template and a different overlay renderer. The archived answers reproduce Table 1; fresh v1 runs are not guaranteed to match those historical numbers.

## Results

Each output directory contains `predictions.jsonl` (answers, raw responses, token usage), `run.json` (settings), `metrics.json` (counts and percentages), and `table1.csv`. A `--limit` run is marked as a subset. Rerun the same command with `--resume` after interruption; saved questions are retained and API errors retried.

To score your own answers on the full test set:

```bash
python evaluate.py score predictions.jsonl --output runs/scored
```

One JSON object per image, e.g. `{"row_id":"salart_ee751cb2d782","q1":"yes","q2":"A","q3":"B","q4":"C"}`. Only whitespace and answer case are normalized. Missing or invalid answers count as wrong. Resumed JSONL files use the last record for each ID.

All Q1–4 means **all four answers on the same image are correct**; All Q2–4 means all three downstream answers are correct. The denominators are 475 artifact, 356 real reference, and 119 generated reference images. Generated references skip Q1, so overall Q1 / All Q1–4 use 831 images; overall All Q2–4 uses 950.

For an offline dataset snapshot, pass `--data-dir /path/to/snapshot` containing `data/*.parquet`. API requests contain only the image, question, and options; labels and provenance are never sent.

## Citation

```bibtex
@misc{sun2026salartvqadiagnosingvlmsunderstand,
  title={SalArt-VQA: Diagnosing Whether VLMs Understand Salient Artifacts in Generated Images},
  author={Xiaoxiao Sun and Ruotian Zhang and Junzhe Huang and James Burgess and Serena Yeung-Levy},
  year={2026},
  eprint={2606.12671},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2606.12671}
}
```

Code: MIT. Dataset and archived annotations: [CC BY 4.0](https://huggingface.co/datasets/salartvqa/SalArt-VQA/blob/main/LICENSE).
