"""CPU-only preflight tests: these must not import torch/unsloth."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmark_qlora.py"
spec = importlib.util.spec_from_file_location("benchmark_qlora", MODULE_PATH)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    def test_text_validation_and_dataset_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "data.jsonl"
            p.write_text(''.join(json.dumps({"text": str(i)}, ensure_ascii=False) + '\n' for i in range(4)))
            self.assertEqual(benchmark.read_texts(p), ["0", "1", "2", "3"])
            self.assertEqual(len(benchmark.dataset_hash(p)), 64)
            p.write_text('{"text":"ok"}\n')
            with self.assertRaisesRegex(ValueError, "four"):
                benchmark.read_texts(p)
            p.write_text('INVALID JSON\n')
            with self.assertRaisesRegex(ValueError, "Invalid JSON"):
                benchmark.read_texts(p)

    def test_compare_rejects_mismatched_training_profiles(self):
        profile = {key: "same" for key in benchmark.PROFILE_KEYS}
        before = {"profile": profile, "metrics": {"train_steps_per_second": 1.0, "peak_gpu_allocated_gb": 10}}
        after = {"profile": {**profile, "dataset_sha256": "changed"}, "metrics": {"train_steps_per_second": 2.0}}
        with self.assertRaisesRegex(ValueError, "dataset_sha256"):
            benchmark.compare_results(before, after)
        after["profile"]["dataset_sha256"] = "same"
        ratios = benchmark.compare_results(before, after)
        self.assertEqual(ratios["train_steps_per_second_ratio_new_to_baseline"], 2.0)
        self.assertNotIn("peak_gpu_allocated_gb_ratio_new_to_baseline", ratios)

    def test_fails_before_importing_gpu_on_bad_local_model(self):
        from argparse import Namespace
        with tempfile.TemporaryDirectory() as folder:
            values = dict(model=folder, data="missing.jsonl", output=folder,
                          steps=5, seq_len=256, batch_size=1, grad_accum=4,
                          lora_rank=8, min_gpu_gb=10.0, save_adapter=False)
            with self.assertRaisesRegex(ValueError, "local Hugging Face"):
                benchmark.run(Namespace(**values))


if __name__ == "__main__":
    unittest.main()
