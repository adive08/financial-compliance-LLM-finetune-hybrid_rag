# Practical 1 — Financial Compliance and Advisory Engine
## Simplified Development Prompt

You are an expert AI engineer and full-stack developer specializing in production-quality LLM applications under severe hardware constraints.

We are building a college project called:

**Financial Compliance and Advisory Engine using Fine-tuned LLMs and Hybrid RAG**

However, this is **Practical 1 of 3**, so DO NOT build the complete production system yet.

The objective of Practical 1 is only to demonstrate that:

1. We have a small financial-compliance-focused language model.
2. The model can be fine-tuned on a small dataset.
3. The trained model/adapters are actually saved to disk.
4. Training can be resumed if it is interrupted.
5. A very small web interface is connected to the trained model.
6. A user can upload a financial document and receive a small, simple analysis/result.
7. The model training itself happens on a **separate laptop**, not on the development machine/server.
8. The complete codebase should be suitable for pushing to a GitHub repository so another team member can clone it and train the model on their laptop.

Do not over-engineer Practical 1. We need a working proof of concept that we can demonstrate.

---

# 1. HARDWARE CONSTRAINTS

The model-training laptop has:

- NVIDIA RTX 3050 Ti
- 4 GB VRAM
- 16 GB system RAM
- 512 GB SSD
- Approximately 3–4 days available for training

Because of this, training MUST be extremely VRAM-efficient.

Use:

- 4-bit quantization
- QLoRA
- LoRA adapters
- Gradient checkpointing
- Very small per-device batch size
- Gradient accumulation
- FP16 where supported
- Unsloth where compatible
- Hugging Face PEFT/Transformers
- Small sequence lengths
- Frequent checkpoints

Do NOT attempt to train a large model.

---

# 2. MODEL FOR PRACTICAL 1

Use a very small model that is realistic for an RTX 3050 Ti 4 GB GPU.

Preferred order:

1. `unsloth/Qwen2.5-0.5B-Instruct` — recommended first choice
2. `unsloth/Qwen2.5-1.5B-Instruct` — use only if the 0.5B model is insufficient and memory allows
3. TinyLlama 1.1B as a fallback

For Practical 1, prioritize **successful training and resumability over model size or quality**.

The model should be fine-tuned to perform a very narrow task such as:

> Given a financial document snippet and a small set of regulatory information, identify whether there appears to be a compliance issue and return a short structured response.

Do not attempt to make the model an authoritative legal advisor.

The application must clearly state that the output is a **prototype/demo and not legal or financial advice**.

---

# 3. TRAINING MUST BE SEPARATE FROM THE APPLICATION

This is extremely important.

The project has two separate environments:

## A. Training machine

The other team member's laptop will be used for:

- Dataset preparation
- Model downloading
- QLoRA fine-tuning
- Checkpoint creation
- Resuming interrupted training
- Saving the final LoRA adapter/model

The training code must NOT require the web application to be running.

The training laptop should produce artifacts such as:

```text
models/
└── financial-compliance-qwen/
    ├── checkpoint-100/
    ├── checkpoint-200/
    ├── checkpoint-300/
    ├── ...
    ├── final/
    └── adapter_config.json
```

Only the `final/` directory (the LoRA adapter, tokenizer files and
`adapter_config.json`) needs to be transferred to the application machine.
Intermediate checkpoints stay on the training laptop.

## B. Application machine

The development machine runs only:

- The web interface
- Document upload and text extraction
- Inference against the trained adapter

It must never attempt to train, and it must start cleanly even when no
trained adapter is present yet (see the fallback rule in section 6).

## Handoff between the two machines

- The trained adapter is small (tens of MB). Transfer it by any convenient
  means: a release attachment, a shared drive, or a zip file.
- Do **not** commit model weights or checkpoints to Git. Add `models/` and
  `*.gguf`, `*.safetensors`, `*.bin` to `.gitignore`.
- Only code, configuration and the small dataset belong in the repository.

---

# 4. RESUMABLE TRAINING

Training may be interrupted (laptop sleep, power loss, thermal shutdown,
a closed lid, or simply stopping for the night). Resuming must be trivial.

Requirements:

- Save a checkpoint every N steps (start with `save_steps=100`).
- Keep a bounded number of checkpoints (`save_total_limit=3`) so the SSD
  does not fill up.
- The training script must accept a `--resume` flag.
- On `--resume`, automatically detect the newest `checkpoint-*` directory
  and continue from it via `trainer.train(resume_from_checkpoint=...)`.
- If no checkpoint exists, start from scratch without error.
- Optimizer state, scheduler state and the RNG seed must be restored, not
  just the weights — otherwise the resumed run is not equivalent.
- Log progress to a file so a crashed run can be diagnosed afterwards.

The expected usage is:

```bash
python train.py                # fresh run
python train.py --resume       # continue after an interruption
```

---

# 5. DATASET FOR PRACTICAL 1

Keep the dataset deliberately small — a few hundred examples is enough to
demonstrate that fine-tuning works.

- Store it in the repository as JSONL so it is diffable and reviewable.
- Use a single, consistent instruction format.
- Include both compliant and non-compliant examples so the model learns
  to distinguish them.

Suggested record shape:

```json
{
  "instruction": "Review the following financial document snippet for compliance issues.",
  "input": "<document snippet>",
  "output": "{\"issue_found\": true, \"category\": \"disclosure\", \"explanation\": \"...\"}"
}
```

The `output` should be a short structured response so the web interface can
parse and display it. Do not aim for long free-form legal reasoning.

---

# 6. WEB INTERFACE

Keep it minimal. A single page is sufficient.

Required behaviour:

- Upload a financial document (PDF or TXT is enough for Practical 1).
- Extract text from the upload.
- Send a snippet of that text to the fine-tuned model.
- Display the short structured analysis returned by the model.
- Display a persistent disclaimer stating that this is a prototype and
  **not legal or financial advice**.

Explicitly out of scope for the interface in Practical 1: user accounts,
databases, history, multi-document comparison, styling beyond the basics.

**Fallback rule:** if no trained adapter is found on disk, the application
must still start and must show a clear message saying the model has not been
trained yet — rather than crashing on import.

---

# 7. REPOSITORY STRUCTURE

The repository must be clonable and runnable by a team member who has
never seen it. Aim for something close to:

```text
.
├── README.md
├── requirements-train.txt      # training machine only (CUDA, bitsandbytes, peft)
├── requirements-app.txt        # application machine only (web + inference)
├── .gitignore                  # ignores models/, checkpoints, venvs, uploads
├── data/
│   └── train.jsonl
├── train/
│   ├── train.py                # QLoRA fine-tune, supports --resume
│   └── config.yaml             # model id, LoRA rank, batch size, save_steps
├── app/
│   ├── main.py                 # web server
│   ├── inference.py            # loads base model + LoRA adapter
│   └── templates/
└── docs/
    └── practical-1-prompt.md   # this document
```

Two separate requirements files matter: the application machine should not
be forced to install CUDA-only packages such as `bitsandbytes`.

---

# 8. SETUP AND RUN INSTRUCTIONS

The README must contain copy-pasteable instructions for both machines.

## Training machine

```bash
git clone <repo-url>
cd <repo>
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements-train.txt
python train/train.py                              # or --resume
```

## Application machine

```bash
pip install -r requirements-app.txt
# copy the trained adapter into models/financial-compliance-qwen/final/
python app/main.py
```

Document the expected VRAM usage and the approximate time per 100 steps so
the training team member knows whether the run is behaving normally.

---

# 9. EXPLICITLY OUT OF SCOPE FOR PRACTICAL 1

Do not build these yet — they belong to Practicals 2 and 3:

- Hybrid RAG (vector search + keyword/BM25 retrieval)
- A regulatory document corpus and its ingestion pipeline
- Citation of specific regulations
- Evaluation harnesses and benchmark scoring
- Authentication, multi-user support, persistence
- Deployment, containerisation, CI/CD
- Any model larger than ~1.5B parameters

If a design decision in Practical 1 would make these harder later, prefer
the option that leaves the door open — but do not implement them now.

---

# 10. ACCEPTANCE CRITERIA

Practical 1 is complete when all of the following can be demonstrated live:

1. `train.py` runs on the 4 GB laptop without an out-of-memory error.
2. Checkpoint directories appear on disk during training.
3. Training is interrupted with Ctrl+C and resumed with `--resume`, and the
   step counter continues from where it stopped rather than restarting.
4. A `final/` adapter directory exists after training.
5. The web interface starts on the application machine.
6. A document is uploaded and a short structured analysis is displayed.
7. The disclaimer is visible on screen.
8. A fresh `git clone` plus the README instructions reproduces the setup.

---

# 11. DISCLAIMER

This system is a student prototype built for coursework. Its output is
generated by a small fine-tuned language model, is not verified against
current regulations, and is **not legal, financial or compliance advice**.
This disclaimer must appear in the README and in the web interface itself.
