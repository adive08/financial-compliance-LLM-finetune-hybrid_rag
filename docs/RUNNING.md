# How to run it

A short runbook. Full detail is in the [README](../README.md).

Two machines, and they never run at the same time:

```text
TRAINING LAPTOP (RTX 3050 Ti)          APPLICATION MACHINE
python train/train.py          ──►     models/.../final/
                                       python scripts/build_rag_index.py
                                       python app/main.py
```

---

## Training laptop

```bash
git clone <repo-url>
cd financial-compliance-LLM-finetune-hybrid_rag

python -m venv .venv
.venv\Scripts\activate                 # Windows
# source .venv/bin/activate            # Linux

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-train.txt
```

Work up in three steps. Do not skip to the third.

```bash
python train/train.py --dry-run        # 1. no GPU. Checks the data only.
python train/train.py --max-steps 20   # 2. ~2 min. Proves the GPU path works.
python train/train.py                  # 3. the real run.
```

A checkpoint is written every 100 steps and the newest three are kept. Progress
also goes to `logs/train.log`.

### Stopping and resuming

`Ctrl+C` is safe at any point.

```bash
python train/train.py --resume
```

**Check the step counter.** It must continue from the saved step, not restart at
0. If no checkpoint exists, `--resume` starts a fresh run instead of failing.

### When it finishes

Copy only this folder to the application machine, to the same path:

```text
models/financial-compliance-qwen/final/
```

Leave the `checkpoint-*` folders behind.

---

## Application machine

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch
pip install -r requirements-app.txt
```

Build the regulatory index (once, and after corpus/config changes):

```bash
python scripts/build_rag_index.py
```

Start the app:

```bash
python app/main.py                     # http://127.0.0.1:8000
```

Use `PORT=8080 python app/main.py` if 8000 is taken. The app starts whether or
not the adapter and index exist; missing pieces are shown as banners.

To demo, upload [sample-document.txt](sample-document.txt) with an as-of date of
`2024-06-01`. It should produce several findings with citations to the 2023 KYC
regulation, the advertising code and the fee disclosure rules.

### Offline / low-dependency mode

If `fastembed` or `qdrant-client` is unavailable, the system falls back to
deterministic hashing embeddings and a numpy cosine store. To force that path:

```bash
python scripts/build_rag_index.py --force --embedder hashing --vector numpy
```

If `langgraph` is unavailable, the same four agent nodes run through a
sequential runner with identical inputs and outputs.

### Single-model fallback

The original Practical 1 path is still available:

```bash
python scripts/run_pipeline.py --file docs/sample-document.txt --mode single
```

or run the whole web app in that mode:

```bash
$env:PIPELINE_MODE="single"; python app/main.py    # PowerShell
PIPELINE_MODE=single python app/main.py            # Linux/macOS
```

---

## Multi-agent CLI

```bash
python scripts/run_pipeline.py --file docs/sample-document.txt
python scripts/run_pipeline.py --file docs/sample-document.txt --json
python scripts/run_pipeline.py --file docs/sample-document.txt --report reports/demo.pdf
```

Temporal check:

```bash
python scripts/run_pipeline.py --file docs/sample-document.txt --as-of 2020-06-01
python scripts/run_pipeline.py --file docs/sample-document.txt --as-of 2024-06-01
```

The first cites `rbi-kyc-md-2016`; the second cites `rbi-kyc-md-2023`.

---

## Evaluation

```bash
python scripts/evaluate.py --limit 10
python scripts/evaluate.py --limit 0 --no-advisory
```

Outputs land in `reports/eval-<timestamp>.json` and `.md`. The evaluation needs
the trained adapter; without it the script stops with a clear message.

---

## Checking things work

```bash
python tests/test_pipeline.py          # original 22 checks
python tests/test_rag.py
python tests/test_agents.py
python tests/test_evaluate.py
python tests/test_report.py
curl http://127.0.0.1:8000/health
```

`/health` reports `adapter_present`, the active pipeline mode, the graph backend
(`langgraph` or `sequential`) and the RAG index status.

---

## When it goes wrong

| Symptom | Fix |
|---|---|
| `No RAG index` banner | `python scripts/build_rag_index.py` |
| Index is stale after editing the corpus | `python scripts/build_rag_index.py --force` |
| `fastembed` cannot download the model | rebuild with `--embedder hashing` |
| Qdrant lock error | stop other processes using `storage/qdrant`, or rebuild with `--vector numpy` |
| PDF download returns 503 | `pip install reportlab` (already in requirements) |
| `CUDA out of memory` | `max_seq_length` to 256 in `train/config.yaml`, then `lora.r` to 8 |
| `No CUDA GPU found` | You are on the wrong machine. Use `--dry-run` to check data anywhere. |
| Resume restarts at step 0 | You omitted `--resume`, or no `checkpoint-*` folders exist. |
| `Model not trained yet` banner | The `final/` folder is missing or in the wrong place. |
| "Unstructured reply" in the UI | A 0.5B model sometimes returns non-JSON; confidence is lowered and the raw text is kept. |
| Scanned PDF rejected | No selectable text. OCR is out of scope. |
| Empty citations on a finding | The index was built before the corpus was updated, or the retrieved regulation category did not match. Rebuild and check the agent trace. |

---

Output is from a small fine-tuned model and a synthetic corpus and is **not
legal, financial or compliance advice.**
