#!/usr/bin/env python3
"""Opt-in, local-only Unsloth QLoRA benchmark for VietCore AI.

Standard-library imports are intentionally the only imports at module import time;
GPU packages are imported only after preflight validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

PROFILE_KEYS = ("model", "dataset_sha256", "steps", "seq_len", "batch_size", "grad_accum", "lora_rank")


def read_texts(path: Path) -> list[str]:
    """Read local JSONL with a non-empty 'text' field. Never upload it."""
    if not path.is_file():
        raise ValueError(f"Dataset not found: {path}")
    texts = []
    with path.open("r", encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {number}") from exc
            if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
                raise ValueError(f"Line {number}: expected nonempty string 'text'")
            texts.append(item["text"].strip())
    if len(texts) < 4:
        raise ValueError("At least four JSONL text records are required (train and evaluation)")
    return texts


def dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_results(baseline: dict, current: dict) -> dict:
    """Refuse comparisons across different workloads (and missing metrics)."""
    left, right = baseline.get("profile", {}), current.get("profile", {})
    mismatched = [key for key in PROFILE_KEYS if not left.get(key) or left.get(key) != right.get(key)]
    if mismatched:
        raise ValueError("Incomparable benchmark profiles: " + ", ".join(mismatched))
    out = {}
    for key in ("train_steps_per_second", "peak_gpu_allocated_gb", "eval_loss"):
        old, new = baseline.get("metrics", {}).get(key), current.get("metrics", {}).get(key)
        if not isinstance(old, (int, float)) or not isinstance(new, (int, float)) or old <= 0 or new < 0:
            continue
        if not math.isfinite(old) or not math.isfinite(new):
            continue
        out[key + "_ratio_new_to_baseline"] = round(new / old, 4)
    return out


def run(args: argparse.Namespace) -> dict:
    if args.steps < 1 or args.steps > 1000 or args.seq_len < 32 or args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("Invalid training parameters")
    if args.lora_rank not in (4, 8, 16, 32):
        raise ValueError("LoRA rank must be one of 4, 8, 16, 32")
    model_dir = Path(args.model).expanduser().resolve()
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise ValueError("--model must point to an already-downloaded local Hugging Face model directory")
    data_path = Path(args.data).expanduser().resolve()
    texts = read_texts(data_path)
    profile = {
        "model": str(model_dir), "dataset_sha256": dataset_hash(data_path),
        "steps": args.steps, "seq_len": args.seq_len, "batch_size": args.batch_size,
        "grad_accum": args.grad_accum, "lora_rank": args.lora_rank,
    }
    # Preflight before importing GPU frameworks or allocating model weights.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA unavailable. This benchmark must run on a CUDA GPU worker")
    available_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if available_gb < args.min_gpu_gb:
        raise RuntimeError(f"GPU has {available_gb:.1f} GiB, minimum required: {args.min_gpu_gb} GiB")

    from unsloth import FastLanguageModel
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir), max_seq_length=args.seq_len, dtype=None, load_in_4bit=True
    )
    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_rank * 2, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    class LocalTextDataset(torch.utils.data.Dataset):
        def __init__(self, values: list[str]):
            self.rows = [tokenizer(value, max_length=args.seq_len, truncation=True) for value in values]

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict:
            return self.rows[index]

    train_data = LocalTextDataset(texts[:-1])
    eval_data = LocalTextDataset(texts[-1:])
    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_args = TrainingArguments(
        output_dir=str(out_dir / "trainer"), per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, max_steps=args.steps,
        learning_rate=2e-4, warmup_steps=0, weight_decay=0.01,
        optim="adamw_torch", fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(), save_strategy="no",
        report_to="none", logging_strategy="no", seed=42,
        dataloader_num_workers=0, dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model, args=train_args, train_dataset=train_data, eval_dataset=eval_data,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    torch.cuda.reset_peak_memory_stats(0)
    start = time.monotonic()
    training = trainer.train()
    train_seconds = time.monotonic() - start
    peak_allocated = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    peak_reserved = torch.cuda.max_memory_reserved(0) / (1024 ** 3)
    evaluation = trainer.evaluate()
    if args.save_adapter:
        model.save_pretrained(str(out_dir / "lora_adapter"))
        tokenizer.save_pretrained(str(out_dir / "lora_adapter"))
    return {
        "schema_version": 1, "engine": "unsloth-qlora-4bit", "profile": profile,
        "gpu": {"name": torch.cuda.get_device_name(0), "capacity_gb": round(available_gb, 3)},
        "metrics": {
            "train_seconds": round(train_seconds, 3),
            "train_steps_per_second": round(args.steps / train_seconds, 5),
            "peak_gpu_allocated_gb": round(peak_allocated, 4),
            "peak_gpu_reserved_gb": round(peak_reserved, 4),
            "train_loss": training.training_loss,
            "eval_loss": evaluation.get("eval_loss"),
        },
        "notes": "Eval loss is a proxy, not a validated task quality/accuracy score. No data was uploaded by this script.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local model directory; remote IDs are rejected")
    parser.add_argument("--data", required=True, help="Local JSONL with text field")
    parser.add_argument("--output", required=True, help="Directory for benchmark.json and optional adapter")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--min-gpu-gb", type=float, default=10.0)
    parser.add_argument("--save-adapter", action="store_true", help="Opt-in save LoRA weights locally")
    parser.add_argument("--baseline-json", help="Previously collected benchmark for an identical workload")
    args = parser.parse_args(argv)
    result = run(args)
    if args.baseline_json:
        baseline = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
        result["comparison"] = compare_results(baseline, result)
    result_path = Path(args.output).expanduser().resolve() / "benchmark.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Benchmark saved to {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
