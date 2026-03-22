import json
import tempfile
import unittest
from pathlib import Path

from arena.comparison import compare_runs
from arena.reporting import to_csv, to_markdown
from arena.run_store import EvaluationResults, load_run, save_run


def row(prompt="p1", **changes):
    value = {"model": "model-a", "prompt_id": prompt, "response": "answer",
             "latency_seconds": 2.0, "judge_score": 0.5, "memory_mb": 100}
    value.update(changes)
    return value


def run(rows, **metadata):
    return {"name": "Example", "rows": rows, "metadata": metadata}


def metric(report, label="Judge score (0–1)"):
    return next(r for r in report["metrics"] if r["metric"] == label)


class ComparisonTests(unittest.TestCase):
    def test_paired_delta_excludes_unmatched_prompt(self):
        report = compare_runs(run([row(), row("extra", judge_score=1)]), run([row(judge_score=0.7)]))
        self.assertAlmostEqual(metric(report)["delta"], 0.2)
        self.assertEqual(metric(report)["paired_samples"], 1)
        self.assertTrue(any("coverage differs" in w for w in report["warnings"]))

    def test_failure_is_counted_but_not_averaged_as_zero(self):
        report = compare_runs(run([row(), row("p2")]), run([row(error="Timeout", response="", judge_score=0), row("p2", judge_score=0.8)]))
        self.assertEqual(report["summaries"][0]["current_failures"], 1)
        self.assertEqual(metric(report)["paired_samples"], 1)
        self.assertAlmostEqual(metric(report)["delta"], 0.3)

    def test_missing_and_nan_metrics_remain_unavailable(self):
        report = compare_runs(run([row(judge_score=None)]), run([row(judge_score=float("nan"))]))
        self.assertIsNone(metric(report)["delta"])
        self.assertEqual(metric(report)["paired_samples"], 0)

    def test_zero_judge_score_is_valid(self):
        report = compare_runs(run([row(judge_score=0)]), run([row(judge_score=0.5)]))
        self.assertEqual(metric(report)["delta"], 0.5)

    def test_zero_memory_is_not_false_improvement(self):
        report = compare_runs(run([row()]), run([row(memory_mb=0)]))
        self.assertIsNone(metric(report, "Memory sample (MB)")["delta"])

    def test_duplicates_do_not_overwrite_each_other(self):
        report = compare_runs(run([row(), row()]), run([row()]))
        self.assertEqual(report["pairs"], [])
        self.assertTrue(any("Duplicate" in w for w in report["warnings"]))

    def test_legacy_runs_warn_about_prompt_identity(self):
        report = compare_runs(run([row()]), run([row()]))
        self.assertTrue(any("matching IDs" in w for w in report["warnings"]))

    def test_changed_prompt_text_warns(self):
        report = compare_runs(run([row()], prompt_set_hash="a"), run([row()], prompt_set_hash="b"))
        self.assertTrue(any("Prompt sets" in w for w in report["warnings"]))

    def test_different_models_have_no_fake_delta(self):
        report = compare_runs(run([row()]), run([row(model="model-b")]))
        self.assertEqual(len(report["summaries"]), 2)
        self.assertTrue(all(m["delta"] is None for m in report["metrics"]))

    def test_all_failed_and_empty_runs_are_supported(self):
        report = compare_runs(run([row(error="offline", response="")]), run([]))
        self.assertEqual(report["summaries"][0]["baseline_failures"], 1)
        self.assertTrue(all(m["delta"] is None for m in report["metrics"]))
        self.assertEqual(compare_runs(run([]), run([]))["metrics"], [])

    def test_report_explains_failures_and_missing_values(self):
        report = compare_runs(run([row(judge_score=None)]), run([row()]))
        text = to_markdown(report)
        self.assertIn("N/A", text)
        self.assertIn("Coverage and failures", text)
        self.assertIn("current − baseline", text)

    def test_csv_escapes_formula_text_but_preserves_numeric_delta(self):
        text = to_csv([{"model": "=BAD()", "delta": -0.5}])
        self.assertIn("'=BAD()", text)
        self.assertIn(",-0.5", text)


class StorageTests(unittest.TestCase):
    def test_roundtrip_keeps_list_format_and_unique_names(self):
        with tempfile.TemporaryDirectory() as directory:
            results = EvaluationResults([row()], {"prompt_set_hash": "abc"})
            first = save_run(results, directory, "Baseline")
            second = save_run(results, directory, "Baseline")
            self.assertNotEqual(first, second)
            self.assertIsInstance(json.loads(Path(first).read_text()), list)
            loaded = load_run(first)
            self.assertEqual(loaded["name"], "Baseline")
            self.assertEqual(loaded["metadata"]["prompt_set_hash"], "abc")
            self.assertEqual(loaded["rows"], [row()])
            self.assertEqual(len(list(Path(directory).glob("eval_*.json"))), 2)

    def test_old_files_load_without_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval_old.json"
            path.write_text(json.dumps([row()]))
            self.assertEqual(load_run(path)["metadata"], {})

    def test_invalid_result_shape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval_invalid.json"
            for data in ({"not": "a list"}, ["bad"], [{"model": "a"}]):
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_run(path)

    def test_broken_metadata_does_not_hide_valid_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(save_run([row()], directory))
            (path.parent / "metadata" / path.name).write_text("{invalid")
            loaded = load_run(path)
            self.assertEqual(loaded["rows"], [row()])
            self.assertTrue(loaded["warnings"])


if __name__ == "__main__":
    unittest.main()
