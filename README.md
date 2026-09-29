# SalArt-VQA

**Diagnosing Whether VLMs Understand Salient Artifacts in Generated Images**

[Paper](https://arxiv.org/abs/2606.12671) · [Dataset](https://huggingface.co/datasets/salartvqa/SalArt-VQA) · [Leaderboard](https://zhangruotian.github.io/SalArt-VQA/)

950 images and 3,681 questions covering artifact detection, region selection, box grounding, and evidence selection.

## Evaluate a model

Requires Python 3.10+.

```bash
git clone https://github.com/zhangruotian/SalArt-VQA.git
cd SalArt-VQA
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY='your-key'
python evaluate.py run --provider openai --model gpt-5.4-nano \
  --output runs/gpt-5.4-nano --limit 1
```

The v1 dataset downloads automatically (~1.13 GB in total). `--limit 1` tests one image; **remove it for the full evaluation**.

To use another service, change `--provider` and `--model`, set its key, and choose a new `--output` directory:

| Provider | Example model | Key environment variable |
|---|---|---|
| `openai` | `gpt-5.4-nano` | `OPENAI_API_KEY` |
| `anthropic` | `claude-opus-4-6` | `ANTHROPIC_API_KEY` |
| `gemini` | `gemini-3.1-pro-preview` | `GEMINI_API_KEY` |
| `kimi` | `kimi-k2.5` | `MOONSHOT_API_KEY` |
| `ollama` | `qwen3-vl:8b` | None locally |
| `vllm` | `Qwen/Qwen3-VL-8B-Instruct` | `VLLM_API_KEY` if enabled |
| `openai-compatible` | Your server's model name | `OPENAI_API_KEY` if enabled |

For local models, start the server separately:

```bash
# Ollama: with `ollama serve` running in another terminal
ollama pull qwen3-vl:8b

# Or vLLM: requires vLLM installed and a suitable GPU
vllm serve Qwen/Qwen3-VL-8B-Instruct --max-model-len 16384
```

Ollama defaults to `http://localhost:11434/v1`, vLLM to `http://localhost:8000/v1`. Override with `--base-url URL`; this flag is required for `openai-compatible`.

- `--workers`: concurrent images (default: 4).
- `--max-tokens`: output budget per question (default: 4,096).
- `--temperature 0`: use when supported; otherwise omit for the provider default.
- `--resume`: continue a saved run with the same command and settings.

See `python evaluate.py run --help` for additional options.

## Results

Results are saved under `--output`: `table1.csv` contains the Table 1 metrics, `metrics.json` includes counts, `predictions.jsonl` stores answers and responses, and `run.json` records settings. Missing or invalid answers count as incorrect.

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

Code: MIT. Dataset: [CC BY 4.0](https://huggingface.co/datasets/salartvqa/SalArt-VQA/blob/main/LICENSE).
