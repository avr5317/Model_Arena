"""Paired raw metrics; never compare independently normalized leaderboards."""
import math
from collections import Counter
from statistics import mean

METRICS = {
    "judge_score": ("Judge score (0–1)", "higher"),
    "keyword_accuracy": ("Keyword score (0–1)", "higher"),
    "bleu1_score": ("Existing overlap score (0–1)", "higher"),
    "tokens_per_second": ("Generation speed (tokens/s)", "higher"),
    "latency_seconds": ("Request latency (s)", "lower"),
    "memory_mb": ("Memory sample (MB)", "lower"),
    "load_seconds": ("Model loading (s)", "lower"),
    "prompt_eval_seconds": ("Prompt processing (s)", "lower"),
    "generation_seconds": ("Generation duration (s)", "lower"),
}


def failed(row):
    return bool(row.get("error")) or not str(row.get("response") or "").strip()


def _key(row):
    return row["model"], str(row["prompt_id"])


def index_rows(rows):
    counts = Counter(_key(row) for row in rows)
    return {_key(row): row for row in rows if counts[_key(row)] == 1}, counts


def _number(row, metric):
    value = row.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    # The original evaluator uses zero when Docker stats cannot be read.
    if metric == "memory_mb" and value == 0:
        return None
    return value


def compare_runs(baseline, current):
    left, left_counts = index_rows(baseline["rows"])
    right, right_counts = index_rows(current["rows"])
    warnings = list(baseline.get("warnings", [])) + list(current.get("warnings", []))
    if any(n > 1 for n in [*left_counts.values(), *right_counts.values()]):
        warnings.append("Duplicate model/prompt IDs are excluded from paired comparisons.")
    common = sorted(left.keys() & right.keys())
    if not common:
        warnings.append("No unique model/prompt pairs overlap; metric differences are unavailable.")
    if set(left_counts) != set(right_counts):
        warnings.append("Model/prompt coverage differs. Deltas use only overlapping unique pairs.")
    a, b = baseline.get("metadata", {}), current.get("metadata", {})
    if not a.get("prompt_set_hash") or not b.get("prompt_set_hash"):
        warnings.append("Legacy or incomplete metadata: matching IDs do not prove matching prompt text.")
    elif a["prompt_set_hash"] != b["prompt_set_hash"]:
        warnings.append("Prompt sets or scoring instructions changed; treat these deltas as exploratory.")
    for key, label in [("host", "Recorded host details"), ("judge_model", "Judge model"),
                       ("generation", "Generation settings"), ("repository_url", "Analyzed repository"),
                       ("evaluator_commit", "Evaluator commit")]:
        if a.get(key) != b.get(key):
            warnings.append(f"{label} differ between runs.")
    if {r.get("judge_model") for r in baseline["rows"]} != {r.get("judge_model") for r in current["rows"]}:
        warnings.append("Recorded response judge models differ between runs.")
    warnings.append("Model digests, complete hardware settings, and random settings are not fully captured; this is not proof of reproducibility or statistical significance.")

    summaries, metrics = [], []
    models = sorted({r["model"] for r in baseline["rows"] + current["rows"]})
    for model in models:
        lrows = [r for r in baseline["rows"] if r["model"] == model]
        rrows = [r for r in current["rows"] if r["model"] == model]
        keys = [key for key in common if key[0] == model]
        successful = [key for key in keys if not failed(left[key]) and not failed(right[key])]
        summaries.append({"model": model, "baseline_attempts": len(lrows),
                          "current_attempts": len(rrows),
                          "baseline_failures": sum(failed(r) for r in lrows),
                          "current_failures": sum(failed(r) for r in rrows),
                          "matched_pairs": len(keys), "both_successful": len(successful)})
        for metric, (label, direction) in METRICS.items():
            pairs = []
            for key in successful:
                x, y = _number(left[key], metric), _number(right[key], metric)
                if x is not None and y is not None:
                    pairs.append((x, y))
            before = mean(p[0] for p in pairs) if pairs else None
            after = mean(p[1] for p in pairs) if pairs else None
            delta = after - before if pairs else None
            metrics.append({"model": model, "metric": label, "preferred": direction,
                            "baseline": before, "current": after, "delta": delta,
                            "paired_samples": len(pairs)})
    return {"baseline": baseline["name"], "current": current["name"],
            "warnings": list(dict.fromkeys(warnings)), "summaries": summaries,
            "metrics": metrics, "pairs": [(key, left[key], right[key]) for key in common]}
