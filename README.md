# Financial Compliance and Advisory Engine — Practical 1

A proof of concept for the coursework project *Financial Compliance and
Advisory Engine using Fine-tuned LLMs and Hybrid RAG*.

**This is Practical 1 of 3.** Its only goal is to show the pipeline end to
end: a small model is fine-tuned with QLoRA on a 4 GB laptop GPU, training can
be interrupted and resumed, the adapter is saved, and a minimal web interface
loads it and analyses an uploaded document. The Hybrid RAG stage, the
regulatory corpus and evaluation come later.

**In a hurry?** [docs/RUNNING.md](docs/RUNNING.md) is the short runbook.

> **Disclaimer.** This is a student prototype. Its output comes from a small
> fine-tuned language model, is not checked against current regulations, and
> is **not legal, financial or compliance advice.**

---

## How the two machines split up

| | Training laptop (RTX 3050 Ti, 4 GB) | Application machine |
|---|---|---|
| Installs | `requirements-train.txt` | `requirements-app.txt` |
| Needs CUDA | yes | no |
| Runs | `train/train.py` | `app/main.py` |
| Produces | `models/financial-compliance-qwen/final/` | the demo UI |

The two never run at the same time and neither imports the other. The only
thing that moves between them is the `final/` adapter folder — a few tens of
MB. **Model weights are gitignored; do not commit them.**

---

## Layout

```text
.
├── data/
│   ├── make_dataset.py     # regenerates the JSONL from templates (seeded)
│   ├── train.jsonl         # 250 examples
│   └── eval.jsonl          # 60 examples
├── train/
│   ├── train.py            # QLoRA fine-tune, --resume, --dry-run
│   └── config.yaml         # every hyperparameter lives here
├── app/
│   ├── main.py             # FastAPI server
│   ├── inference.py        # base model + LoRA adapter, lazily loaded
│   ├── extract.py          # PDF / TXT text extraction
│   └── templates/index.html
├── shared/
│   └── prompt_format.py    # ONE prompt definition, used by both sides
├── tests/test_pipeline.py  # no GPU or trained model required
└── docs/                   # the Practical 1 spec
```

`shared/prompt_format.py` matters more than it looks. If training and
inference build prompts even slightly differently, the model looks like it
learned nothing and the cause is very hard to spot. Both sides import it.

---

## Training laptop

```bash
git clone <repo-url>
cd financial-compliance-LLM-finetune-hybrid_rag

python -m venv .venv
.venv\Scripts\activate                 # Windows
# source .venv/bin/activate            # Linux

# PyTorch first, matched to your CUDA version:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-train.txt

python train/train.py
```

Check the data pipeline before committing to a long run — this needs no GPU
and prints exactly what the model will be trained on:

```bash
python train/train.py --dry-run
```

### Interrupting and resuming

Press `Ctrl+C` at any point. Checkpoints are written every 100 steps to
`models/financial-compliance-qwen/`, and the three most recent are kept.

```bash
python train/train.py --resume
```

This finds the highest-numbered `checkpoint-N` and restores the optimizer,
the scheduler and the RNG state, not just the weights — so the resumed run is
equivalent to one that never stopped. Watch the step counter: it should
continue from where it stopped, **not** restart at 0. If no checkpoint exists,
`--resume` starts a fresh run rather than failing.

### Settings that matter on 4 GB

All in `train/config.yaml`. Defaults are deliberately conservative:

| Setting | Value | Why |
|---|---|---|
| `max_seq_length` | 512 | VRAM scales with this; the data needs ~174 |
| `per_device_train_batch_size` | 1 | anything higher risks OOM |
| `gradient_accumulation_steps` | 16 | effective batch of 16 without the memory |
| `load_in_4bit` / `nf4` | on | the base model fits in well under 1 GB |
| `gradient_checkpointing` | on | trades compute for memory |
| `optim` | `paged_adamw_8bit` | paged optimiser survives VRAM spikes |
| `save_steps` | 100 | how much work an interruption can cost you |
| `save_total_limit` | 3 | bounded disk use |

**If you hit CUDA out-of-memory:** lower `max_seq_length` to 256 first, then
`lora.r` to 8. Only after that consider a smaller base model. Close anything
using the GPU — a browser can hold several hundred MB.

Qwen2.5-0.5B is the default. `train/config.yaml` also accepts
`Qwen/Qwen2.5-1.5B-Instruct` or `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, but for
Practical 1 a run that finishes beats a bigger model that OOMs.

### Handing the model over

Copy **only** this folder to the application machine, to the same path:

```text
models/financial-compliance-qwen/final/
```

Intermediate checkpoints stay on the training laptop.

---

## Application machine

```bash
pip install -r requirements-app.txt
python app/main.py                     # http://127.0.0.1:8000
```

No `bitsandbytes` and no CUDA — inference runs in fp16 on GPU or Apple
Silicon, fp32 on CPU. The 0.5B model is comfortable on CPU.

The app **starts whether or not the model has been trained.** With no adapter
present it serves the page, still accepts uploads and still extracts text, and
shows a banner explaining what is missing. That way the UI can be built and
demonstrated while training is still running on the other laptop.

Upload a PDF, TXT, MD or CSV file. The first ~1200 characters are sent to the
model and the structured verdict is displayed, alongside the exact snippet
that was analysed. Whole-document chunking arrives with the Hybrid RAG stage.

---

## Tests

```bash
python tests/test_pipeline.py
```

No GPU, no trained model, no pytest. Covers checkpoint discovery (what
`--resume` relies on), tolerance of malformed model output, and the web
routes including upload failure paths.

---

## The dataset

`data/train.jsonl` is **synthetic**, generated from templates by
`data/make_dataset.py` with a fixed seed. 250 training and 60 eval examples
across seven violation categories plus compliant examples, split roughly 70/30
issue-to-clean.

It is sized to prove the fine-tuning pipeline works, not to teach real
compliance judgement. Replace it with a real corpus when one is available —
nothing else has to change, as long as the record shape is kept:

```json
{"instruction": "...", "input": "<document snippet>", "output": "{\"issue_found\": true, ...}"}
```

---

## Deliberately out of scope for Practical 1

Hybrid RAG and vector search, a regulatory corpus, citing specific
regulations, evaluation and benchmarking, authentication, persistence,
multi-document comparison, deployment, and any model beyond ~1.5B parameters.

## Acceptance checklist

- [ ] `python train/train.py --dry-run` prints the data stats
- [ ] `python train/train.py` runs on the 4 GB laptop without OOM
- [ ] `checkpoint-*` folders appear during training
- [ ] `Ctrl+C`, then `--resume`, continues from the saved step
- [ ] `models/financial-compliance-qwen/final/` exists when training ends
- [ ] `python app/main.py` starts on the application machine
- [ ] an uploaded document produces a structured verdict
- [ ] the disclaimer is visible on screen
- [ ] `python tests/test_pipeline.py` passes on a fresh clone
