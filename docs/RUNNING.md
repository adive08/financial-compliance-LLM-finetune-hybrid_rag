# How to run it

A short runbook. Full detail is in the [README](../README.md).

Two machines, and they never run at the same time:

```text
TRAINING LAPTOP (RTX 3050 Ti)          APPLICATION MACHINE
python train/train.py          ──►     models/.../final/  ──►  python app/main.py
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

Then work up in three steps. Do not skip to the third.

```bash
python train/train.py --dry-run        # 1. no GPU. Checks the data only.
python train/train.py --max-steps 20   # 2. ~2 min. Proves the GPU path works.
python train/train.py                  # 3. the real run.
```

Step 1 should print roughly:

```text
examples          : 250
token length      : min 148 / median 163 / max 174 (cap 512)
truncated at cap  : 0
```

Step 3 writes a checkpoint every 100 steps and keeps the newest three.
Progress also goes to `logs/train.log`.

### Stopping and resuming

`Ctrl+C` is safe at any point.

```bash
python train/train.py --resume
```

**Check the step counter.** It must continue from the saved step, not restart
at 0 — that is the whole point of the exercise. If no checkpoint exists yet,
`--resume` quietly starts a fresh run instead of failing.

### When it finishes

Copy this one folder to the application machine, to the same path:

```text
models/financial-compliance-qwen/final/
```

Nothing else moves between the machines. Leave the `checkpoint-*` folders
behind.

---

## Application machine

```bash
pip install -r requirements-app.txt
python app/main.py
```

Open <http://127.0.0.1:8000>. Use `PORT=8080 python app/main.py` if 8000 is
taken.

The app starts **whether or not the model has been trained** — with no
adapter it shows a banner saying so, and uploads still work. So you can build
and demo the UI while training is still running elsewhere.

To demo: upload [sample-document.txt](sample-document.txt), which contains a
guaranteed-return claim the model should flag.

---

## Checking things work

```bash
python tests/test_pipeline.py          # 22 checks, no GPU or model needed
curl http://127.0.0.1:8000/health      # {"status":"ok","adapter_present":true,...}
```

`adapter_present: false` means the app cannot find
`models/financial-compliance-qwen/final/`. Check the copy step above.

---

## When it goes wrong

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | `max_seq_length` to 256 in `train/config.yaml`, then `lora.r` to 8. Close your browser — it can hold hundreds of MB of VRAM. |
| `No CUDA GPU found` | You are on the wrong machine. Use `--dry-run` to check data anywhere. |
| Resume restarts at step 0 | You omitted `--resume`, or `models/financial-compliance-qwen/` has no `checkpoint-*` folders. |
| `Model not trained yet` banner | The `final/` folder is missing or in the wrong place. |
| "Unstructured reply" in the UI | The model returned something that isn't JSON. Expected occasionally from a 0.5B model; the raw text is shown instead. |
| Scanned PDF rejected | No selectable text. OCR is out of scope for Practical 1 — use a text PDF. |

---

Output is from a small fine-tuned model and is **not legal, financial or
compliance advice.**
