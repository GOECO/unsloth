# VietCore AI – Local QLoRA benchmark (experimental)

This is an **additive** adapter in GOECO/unsloth; it does not replace Unsloth core, change the 21 VietCore agents or deploy a service. It produces local GPU measurements to decide whether a specific model/workload fits a 12 GiB NVIDIA card.

## Licensing and security gate

- Upstream Python core declares **Apache-2.0**; repository also contains an **AGPL-3.0** COPYING and AGPL-marked Studio components. Do **not** assume all Studio code can be embedded into closed-source SaaS. Review file-level headers, dependencies and model/data licenses before distribution or network service use.
- No automatic model download, API call or dataset upload is performed by the benchmark script. Use a model directory with config.json already stored locally and locally authorized data. Dependencies can behave differently; block outbound network on isolated GPU workers handling private data.
- Do not publish generated adapters, private datasets, credentials or benchmark outputs containing machine paths. Untrusted model code is not enabled.
- Upstream drift is managed in a separate scoped PR; do not auto-sync thousands of commits to main.

## CPU preflight (no GPU or third-party packages required)

    python -m py_compile vietcore/benchmark_qlora.py
    python -m unittest discover -s vietcore/tests -v

## 12 GiB GPU pilot

Prepare a **local** compatible instruction model (e.g. an appropriately licensed Qwen-family 0.5B–1.5B model) and a JSONL training set with at least four {"text": "..."} records. The last record is held out for an evaluation-loss proxy. A real quality claim needs a representative separate test set and domain-specific metrics. Install the matching Unsloth/PyTorch/CUDA stack on an isolated NVIDIA worker after reviewing dependency installers.

    python vietcore/benchmark_qlora.py --model /models/qwen-local \
      --data /data/approved-sample.jsonl --output /benchmarks/vietcore-run-1 \
      --steps 5 --seq-len 256 --batch-size 1 --grad-accum 4 --lora-rank 8

Results: training wall time, steps/sec, peak CUDA allocated/reserved GiB, training loss and eval loss. **Evaluation loss is not task accuracy**. Saving model adapters is strictly opt-in via --save-adapter.

To compare repeat runs with identical model, dataset hash and hyperparameters:

    python vietcore/benchmark_qlora.py --model /models/qwen-local \
      --data /data/approved-sample.jsonl --output /benchmarks/vietcore-run-2 \
      --baseline-json /benchmarks/vietcore-run-1/benchmark.json

Comparison rejects mismatched workloads. A claim of "2x faster" or "70% VRAM savings" requires a separately measured, equivalently configured baseline. Record versions of torch, transformers, unsloth, GPU driver and CUDA when reporting metrics.

## Acceptance gates

1. **Build**: Python 3.11/3.12 syntax, no heavyweight imports at module import time.
2. **Test**: profile comparator, local data parser and safe model-path preflight pass in CPU CI.
3. **GPU pilot**: fine-tune a licensed local model and save benchmark.json without OOM or transmitting training data externally.
4. **Review**: code, model, dataset and dependency license checks; design authentication, quotas, encryption and tenant isolation before SaaS integration.
5. **Release**: manual approval only; no auto-merge or production deployment.
