"""QLoRA fine-tune for Practical 1, sized for a 4 GB RTX 3050 Ti.

    python train/train.py                 # fresh run
    python train/train.py --resume        # continue from the newest checkpoint
    python train/train.py --dry-run       # inspect the data pipeline, no GPU

Interrupting with Ctrl+C is safe. The most recent checkpoint on disk is kept
and --resume picks up from it, restoring optimizer, scheduler and RNG state.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from shared.prompt_format import build_messages  # noqa: E402

LOG = logging.getLogger("train")


# --------------------------------------------------------------------------
# config + logging
# --------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def setup_logging() -> None:
    log_dir = os.path.join(REPO_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            # append, so an interrupted-and-resumed run keeps one history
            logging.FileHandler(os.path.join(log_dir, "train.log"), encoding="utf-8"),
        ],
    )
    # Keep the training log readable: these libraries are chatty at INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def read_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
    if not records:
        raise ValueError(f"{path} contains no examples. Run: python data/make_dataset.py")
    return records


def encode(record: dict, tokenizer, max_len: int) -> dict:
    """Tokenize one record, masking the prompt so loss applies to the answer.

    Returns input_ids and labels, where every prompt position is -100.
    """
    messages = build_messages(record["input"], record.get("instruction"))
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(
        record["output"] + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]

    # Never sacrifice the answer: trim the prompt instead, keeping its tail so
    # the chat template's generation marker survives.
    budget = max_len - len(answer_ids)
    if budget < 1:
        answer_ids = answer_ids[: max_len - 1]
        budget = 1
    if len(prompt_ids) > budget:
        prompt_ids = prompt_ids[-budget:]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {"input_ids": input_ids, "labels": labels}


class JsonlDataset:
    """Minimal map-style dataset; avoids a hard dependency on `datasets`."""

    def __init__(self, path: str, tokenizer, max_len: int):
        self.records = read_jsonl(path)
        self.examples = [encode(r, tokenizer, max_len) for r in self.records]
        self.truncated = sum(1 for e in self.examples if len(e["input_ids"]) >= max_len)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


class Collator:
    """Right-pads a batch; pads labels with -100 and builds the attention mask."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        width = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention = [], [], []
        for f in features:
            pad = width - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attention.append([1] * len(f["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------

def find_latest_checkpoint(output_dir: str) -> str | None:
    """Newest `checkpoint-N` in output_dir, compared by N numerically."""
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [
        d for d in os.listdir(output_dir)
        if re.fullmatch(r"checkpoint-\d+", d)
        and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None
    newest = max(checkpoints, key=lambda d: int(d.split("-")[1]))
    return os.path.join(output_dir, newest)


def build_training_args(cfg: dict, output_dir: str, has_eval: bool):
    """TrainingArguments, tolerating the eval_strategy rename across versions."""
    from transformers import TrainingArguments

    t = cfg["training"]
    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        optim=t["optim"],
        fp16=t["fp16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        save_strategy="steps",
        seed=t["seed"],
        report_to="none",
        save_safetensors=True,
    )
    if has_eval:
        kwargs["eval_steps"] = cfg["data"]["eval_steps"]

    strategy = "steps" if has_eval else "no"
    try:  # transformers >= 4.41
        return TrainingArguments(eval_strategy=strategy, **kwargs)
    except TypeError:  # older releases
        return TrainingArguments(evaluation_strategy=strategy, **kwargs)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def run_dry_run(cfg: dict, tokenizer_id: str) -> int:
    """Build and inspect the tokenized dataset without touching the GPU."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_len = cfg["model"]["max_seq_length"]
    dataset = JsonlDataset(os.path.join(REPO_ROOT, cfg["data"]["train_file"]),
                           tokenizer, max_len)

    lengths = sorted(len(e["input_ids"]) for e in dataset.examples)
    supervised = [sum(1 for x in e["labels"] if x != -100) for e in dataset.examples]
    LOG.info("examples          : %d", len(dataset))
    LOG.info("token length      : min %d / median %d / max %d (cap %d)",
             lengths[0], lengths[len(lengths) // 2], lengths[-1], max_len)
    LOG.info("truncated at cap  : %d", dataset.truncated)
    LOG.info("supervised tokens : median %d per example",
             sorted(supervised)[len(supervised) // 2])

    example = dataset.examples[0]
    prompt_part = [i for i, l in enumerate(example["labels"]) if l == -100]
    LOG.info("--- example 0, prompt (masked, no loss) ---\n%s",
             tokenizer.decode(example["input_ids"][: len(prompt_part)]))
    LOG.info("--- example 0, answer (supervised) ---\n%s",
             tokenizer.decode(example["input_ids"][len(prompt_part):]))
    LOG.info("dry run OK — data pipeline is sound.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "train", "config.yaml"))
    parser.add_argument("--resume", action="store_true",
                        help="continue from the newest checkpoint in output_dir")
    parser.add_argument("--dry-run", action="store_true",
                        help="tokenize and inspect the dataset, then exit")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="stop after N steps (smoke test)")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    if args.dry_run:
        return run_dry_run(cfg, cfg["model"]["base_id"])

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if not torch.cuda.is_available():
        LOG.error("No CUDA GPU found. 4-bit QLoRA training requires an NVIDIA GPU.")
        LOG.error("Run this on the training laptop. To check the data pipeline "
                  "anywhere, use:  python train/train.py --dry-run")
        return 1

    LOG.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0),
             torch.cuda.get_device_properties(0).total_memory / 1e9)

    output_dir = os.path.join(REPO_ROOT, cfg["training"]["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    # ---- resume decision, made before anything expensive loads ----
    resume_from = None
    if args.resume:
        resume_from = find_latest_checkpoint(output_dir)
        if resume_from:
            LOG.info("Resuming from %s", resume_from)
        else:
            LOG.info("--resume given but no checkpoint found in %s; "
                     "starting a fresh run.", output_dir)

    # ---- tokenizer + data ----
    base_id = cfg["model"]["base_id"]
    tokenizer = AutoTokenizer.from_pretrained(
        base_id, trust_remote_code=cfg["model"]["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_len = cfg["model"]["max_seq_length"]
    train_ds = JsonlDataset(os.path.join(REPO_ROOT, cfg["data"]["train_file"]),
                            tokenizer, max_len)
    eval_path = os.path.join(REPO_ROOT, cfg["data"]["eval_file"])
    eval_ds = (JsonlDataset(eval_path, tokenizer, max_len)
               if os.path.exists(eval_path) else None)
    LOG.info("train examples: %d | eval examples: %d | truncated: %d",
             len(train_ds), len(eval_ds) if eval_ds else 0, train_ds.truncated)

    # ---- 4-bit base model ----
    q = cfg["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=cfg["model"]["trust_remote_code"],
    )
    model.config.use_cache = False          # incompatible with checkpointing
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"])

    lora = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    training_args = build_training_args(cfg, output_dir, has_eval=eval_ds is not None)
    if args.max_steps:
        training_args.max_steps = args.max_steps

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=Collator(tokenizer.pad_token_id),
    )

    try:
        trainer.train(resume_from_checkpoint=resume_from)
    except KeyboardInterrupt:
        LOG.warning("Interrupted. Newest checkpoint: %s", find_latest_checkpoint(output_dir))
        LOG.warning("Continue with:  python train/train.py --resume")
        return 130

    # ---- final artifact: this is what goes to the application machine ----
    final_dir = os.path.join(output_dir, "final")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    LOG.info("Saved adapter to %s", final_dir)
    LOG.info("Copy that folder to the application machine and run: python app/main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
