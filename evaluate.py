import json
import os
import re
import subprocess
import time
from collections import Counter
from datetime import datetime

import requests
from dotenv import load_dotenv

from claude_judge import ClaudeJudge
from arena.run_store import EvaluationResults, capture_metadata, save_run

load_dotenv()

_judge = ClaudeJudge() if os.environ.get("ANTHROPIC_API_KEY") else None

MODELS = {
    "phi3:mini": {"url": "http://localhost:11434", "container": "ollama-phi3"},
    "llama3:8b": {"url": "http://localhost:11435", "container": "ollama-llama3"},
    "mistral:7b": {"url": "http://localhost:11436", "container": "ollama-mistral"},
}


def set_models(models_dict: dict) -> None:
    """Replace the MODELS registry at runtime (used by dynamic pipeline)."""
    global MODELS
    MODELS = models_dict

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")


def load_prompts():
    with open(PROMPTS_FILE) as f:
        return json.load(f)


def wait_for_models(timeout=600):
    """Poll each container until its target model is available."""
    start = time.time()
    pending = dict(MODELS)
    while pending and (time.time() - start) < timeout:
        for model_name, info in list(pending.items()):
            try:
                resp = requests.get(f"{info['url']}/api/tags", timeout=5)
                if resp.ok:
                    tags = resp.json().get("models", [])
                    names = [m.get("name", "") for m in tags]
                    if any(model_name in n for n in names):
                        print(f"  [OK] {model_name} is ready")
                        del pending[model_name]
            except requests.exceptions.ConnectionError:
                pass
        if pending:
            time.sleep(5)
    if pending:
        print(f"  [WARN] Timed out waiting for: {list(pending.keys())}")
    return list(set(MODELS.keys()) - set(pending.keys()))


def get_container_stats(container_name):
    """Get CPU% and memory usage from docker stats."""
    try:
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format",
             "{{.CPUPerc}},{{.MemUsage}}", container_name],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            cpu = float(parts[0].replace("%", ""))
            mem_str = parts[1].split("/")[0].strip()
            mem_mb = 0.0
            if "GiB" in mem_str:
                mem_mb = float(mem_str.replace("GiB", "")) * 1024
            elif "MiB" in mem_str:
                mem_mb = float(mem_str.replace("MiB", ""))
            return {"cpu_percent": cpu, "memory_mb": round(mem_mb, 1)}
    except Exception:
        pass
    return {"cpu_percent": 0.0, "memory_mb": 0.0}


def run_inference(model_name, prompt_text, max_tokens=200):
    """Send a prompt to an Ollama container and measure performance."""
    info = MODELS[model_name]
    url = f"{info['url']}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "stream": False,
        "options": {"num_predict": max_tokens}
    }

    start_time = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=120)
        elapsed = time.perf_counter() - start_time

        if not resp.ok:
            return {"error": f"HTTP {resp.status_code}", "latency_seconds": elapsed}

        data = resp.json()
        response_text = data.get("response", "")
        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 1)
        tokens_per_sec = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0

        stats = get_container_stats(info["container"])

        return {
            "response": response_text,
            "latency_seconds": round(elapsed, 3),
            "tokens_per_second": round(tokens_per_sec, 2),
            "token_count": eval_count,
            "load_seconds": data.get("load_duration", 0) / 1e9 if "load_duration" in data else None,
            "prompt_eval_seconds": data.get("prompt_eval_duration", 0) / 1e9 if "prompt_eval_duration" in data else None,
            "generation_seconds": data.get("eval_duration", 0) / 1e9 if "eval_duration" in data else None,
            "cpu_percent": stats["cpu_percent"],
            "memory_mb": stats["memory_mb"],
        }
    except requests.exceptions.Timeout:
        return {"error": "Timeout", "latency_seconds": 120.0}
    except Exception as e:
        return {"error": str(e), "latency_seconds": time.perf_counter() - start_time}


def compute_bleu1(reference, candidate):
    """Compute unigram BLEU score (precision with brevity penalty)."""
    def tokenize(text):
        return re.findall(r'\w+', text.lower())

    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    if not cand_tokens or not ref_tokens:
        return 0.0

    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)

    clipped = sum(min(cand_counts[w], ref_counts[w]) for w in cand_counts)
    precision = clipped / len(cand_tokens) if cand_tokens else 0

    bp = min(1.0, len(cand_tokens) / len(ref_tokens)) if ref_tokens else 0
    return round(bp * precision, 4)


def compute_keyword_score(expected_keywords, response):
    """Fraction of expected keywords found in the response."""
    if not expected_keywords:
        return 1.0
    response_lower = response.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
    return round(found / len(expected_keywords), 4)


def compute_negated_penalty(negated_keywords, response):
    """Returns a penalty [0, 1] based on negated keywords found. 0 = none found (good)."""
    if not negated_keywords:
        return 0.0
    response_lower = response.lower()
    found = sum(1 for kw in negated_keywords if kw.lower() in response_lower)
    return round(found / len(negated_keywords), 4)


def score_response(prompt_data, response_text):
    """Route scoring based on the prompt's scoring_method field."""
    method = prompt_data.get("scoring_method", "keyword")

    if method == "coherence_only":
        return {"keyword_accuracy": 1.0, "bleu1_score": 0.0}

    if method == "exact":
        ref = prompt_data.get("reference_answer", "").strip().lower()
        matched = float(ref in response_text.lower()) if ref else 0.0
        penalty = compute_negated_penalty(prompt_data.get("negated_keywords", []), response_text)
        return {"keyword_accuracy": max(0.0, round(matched - penalty, 4)), "bleu1_score": 0.0}

    # Default: keyword + BLEU
    kw = compute_keyword_score(prompt_data.get("expected_keywords", []), response_text)
    penalty = compute_negated_penalty(prompt_data.get("negated_keywords", []), response_text)
    kw = max(0.0, round(kw - penalty * 0.5, 4))
    bl = compute_bleu1(prompt_data.get("reference_answer", ""), response_text)
    return {"keyword_accuracy": kw, "bleu1_score": bl}


def evaluate_single(model_name, prompt_data):
    """Evaluate a single model on a single prompt."""
    result = run_inference(model_name, prompt_data["prompt"], prompt_data.get("max_tokens", 200))

    if "error" in result:
        return {
            "prompt_id": prompt_data["id"],
            "category": prompt_data["category"],
            "difficulty": prompt_data.get("difficulty", "unknown"),
            "scoring_method": prompt_data.get("scoring_method", "keyword"),
            "model": model_name,
            "error": result["error"],
            "latency_seconds": result.get("latency_seconds", 0),
            "tokens_per_second": 0,
            "token_count": 0,
            "keyword_accuracy": 0,
            "bleu1_score": 0,
            "cpu_percent": 0,
            "memory_mb": 0,
            "judge_score": None,
            "judge_explanation": "",
            "judge_rubric_breakdown": {},
            "judge_model": None,
            "response": "",
            "timestamp": datetime.now().isoformat(),
        }

    response_text = result["response"]

    # Syed's scoring dispatcher (handles coherence_only, exact, keyword+negated)
    scores = score_response(prompt_data, response_text)

    # Igor's Claude-as-Judge
    category = prompt_data["category"]
    prompt = prompt_data["prompt"]
    reference_answer = prompt_data.get("reference_answer", "")
    rubric_hints = prompt_data.get("rubric_hints", "")

    if _judge:
        judge_result = _judge.evaluate_response(category, prompt, response_text, reference_answer, rubric_hints)
    else:
        judge_result = {"score": None, "explanation": "Judge unavailable", "rubric_breakdown": {}}

    return {
        "prompt_id": prompt_data["id"],
        "category": category,
        "difficulty": prompt_data.get("difficulty", "unknown"),
        "scoring_method": prompt_data.get("scoring_method", "keyword"),
        "model": model_name,
        "response": response_text,
        "latency_seconds": result["latency_seconds"],
        "tokens_per_second": result["tokens_per_second"],
        "token_count": result["token_count"],
        "load_seconds": result.get("load_seconds"),
        "prompt_eval_seconds": result.get("prompt_eval_seconds"),
        "generation_seconds": result.get("generation_seconds"),
        "keyword_accuracy": scores["keyword_accuracy"],
        "bleu1_score": scores["bleu1_score"],
        "cpu_percent": result["cpu_percent"],
        "memory_mb": result["memory_mb"],
        "judge_score": judge_result["score"],
        "judge_explanation": judge_result["explanation"],
        "judge_rubric_breakdown": judge_result["rubric_breakdown"],
        "judge_model": _judge.model if _judge else None,
        "timestamp": datetime.now().isoformat(),
    }


def evaluate_all(available_models=None, progress_callback=None, categories=None):
    """Run all prompts against all available models. Optionally filter by categories."""
    prompts = load_prompts()
    if categories:
        prompts = [p for p in prompts if p.get("category", "").lower() in [c.lower() for c in categories]]
    if available_models is None:
        available_models = list(MODELS.keys())

    # Snapshot the actual filtered prompt set before inference begins.
    metadata = capture_metadata(
        prompts, available_models, getattr(_judge, "model", None),
        os.path.dirname(os.path.abspath(__file__)),
    )
    results = EvaluationResults(metadata=metadata)
    total = len(prompts) * len(available_models)
    done = 0

    for prompt_data in prompts:
        for model in available_models:
            result = evaluate_single(model, prompt_data)
            results.append(result)
            done += 1
            status = "OK" if not result.get("error") else result.get("error", "")
            print(f"  [{done}/{total}] {model} | {prompt_data['id']} | {status}")
            if progress_callback:
                progress_callback(done, total, model, prompt_data["id"])

    return results


def save_results(results, run_name=None, repository_url=None):
    """Save the original result-list format plus optional run metadata."""
    filepath = save_run(results, RESULTS_DIR, run_name, repository_url)
    print(f"  Results saved to {filepath}")
    return filepath


def main():
    print("Model Arena — Evaluation Pipeline")
    print("=" * 40)

    print("\n1. Waiting for models to be ready...")
    available = wait_for_models(timeout=600)

    if not available:
        print("No models available. Exiting.")
        return

    print(f"\n2. Running evaluation on {len(available)} models...")
    results = evaluate_all(available)

    print(f"\n3. Saving results...")
    save_results(results)

    print("\nDone! Run 'streamlit run dashboard.py' to view results.")


if __name__ == "__main__":
    main()