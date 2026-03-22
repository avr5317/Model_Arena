"""Offline integration tests: no model downloads, Docker, or paid API calls."""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from arena.run_store import load_run


class EvaluatorIntegrationTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        requests = types.ModuleType("requests")
        requests.post = Mock()
        requests.exceptions = types.SimpleNamespace(Timeout=TimeoutError, ConnectionError=ConnectionError)
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda: None
        judge = types.ModuleType("claude_judge")
        judge.ClaudeJudge = Mock()
        with patch.dict(sys.modules, {"requests": requests, "dotenv": dotenv, "claude_judge": judge}), patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            spec = importlib.util.spec_from_file_location("test_evaluate", root / "evaluate.py")
            self.evaluator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.evaluator)

    def test_filtered_run_captures_exact_prompts_and_persists_metadata(self):
        e = self.evaluator
        prompts = [{"id": "p1", "category": "Coding", "prompt": "Hello", "max_tokens": 25},
                   {"id": "p2", "category": "Reasoning", "prompt": "Other"}]
        with tempfile.TemporaryDirectory() as directory, patch.object(e, "load_prompts", return_value=prompts), patch.object(e, "evaluate_single", return_value={"model": "m", "prompt_id": "p1", "response": "answer"}), patch.object(e, "RESULTS_DIR", directory):
            callback = Mock()
            results = e.evaluate_all(["m"], callback, categories=["coding"])
            self.assertIsInstance(results, list)
            self.assertEqual(results.metadata["prompts"], [prompts[0]])
            callback.assert_called_once_with(1, 1, "m", "p1")
            path = e.save_results(results, "First run", "https://github.com/example/repo")
            self.assertIsInstance(json.loads(Path(path).read_text()), list)
            loaded = load_run(path)
            self.assertEqual(loaded["name"], "First run")
            self.assertEqual(loaded["metadata"]["repository_url"], "https://github.com/example/repo")

    def test_inference_timing_conversion_and_missing_fields(self):
        e = self.evaluator
        e.set_models({"m": {"url": "http://localhost:11434", "container": "test"}})
        response = Mock(ok=True)
        response.json.return_value = {"response": "answer", "eval_count": 20, "eval_duration": 2_000_000_000, "load_duration": 500_000_000}
        e.requests.post.return_value = response
        with patch.object(e, "get_container_stats", return_value={"cpu_percent": 1, "memory_mb": 100}):
            result = e.run_inference("m", "test")
        self.assertEqual(result["tokens_per_second"], 10)
        self.assertEqual(result["load_seconds"], 0.5)
        self.assertEqual(result["generation_seconds"], 2)
        self.assertIsNone(result["prompt_eval_seconds"])

    def test_timing_fields_reach_evaluation_rows_without_judge(self):
        e = self.evaluator
        inference = {"response": "answer", "latency_seconds": 1, "tokens_per_second": 10,
                     "token_count": 10, "cpu_percent": 1, "memory_mb": 100,
                     "load_seconds": 0.2, "prompt_eval_seconds": 0.1, "generation_seconds": 0.7}
        with patch.object(e, "run_inference", return_value=inference):
            row = e.evaluate_single("m", {"id": "p", "category": "Coding", "prompt": "test"})
        self.assertEqual(row["generation_seconds"], 0.7)
        self.assertIsNone(row["judge_score"])


if __name__ == "__main__":
    unittest.main()
