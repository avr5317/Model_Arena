"""
Model Arena MCP Server (Combined)

Exposes evaluation pipeline + Claude judge + prompt management as MCP tools.
Uses Syed's clean FastMCP architecture with Igor's judge integration.

Register in Claude Code via .claude/settings.json:
  "mcpServers": {
    "model-arena": {
      "command": "python3",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/project"
    }
  }
"""

import json
import os
import glob as glob_mod
import uuid

import pandas as pd
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

from evaluate import (
    load_prompts,
    evaluate_all,
    save_results,
    wait_for_models,
    set_models,
    PROMPTS_FILE,
    RESULTS_DIR,
    MODELS,
)
from claude_judge import ClaudeJudge
from repo_analyzer import analyze_github_url
from model_discovery import search_models
from container_manager import setup_containers

mcp = FastMCP("model-arena")

_judge = ClaudeJudge() if os.environ.get("ANTHROPIC_API_KEY") else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_prompts(prompts: list) -> None:
    with open(PROMPTS_FILE, "w") as f:
        json.dump(prompts, f, indent=2)


def _load_latest_results() -> list | None:
    if not os.path.exists(RESULTS_DIR):
        return None
    files = sorted(glob_mod.glob(os.path.join(RESULTS_DIR, "eval_*.json")))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def _compute_leaderboard(results: list) -> list:
    """Return composite-scored leaderboard from raw results."""
    df = pd.DataFrame(results)
    df_valid = df[df["response"].astype(str).str.len() > 0]
    if df_valid.empty:
        return []

    stats = df_valid.groupby("model").agg(
        avg_tokens_per_second=("tokens_per_second", "mean"),
        avg_keyword_accuracy=("keyword_accuracy", "mean"),
        avg_bleu=("bleu1_score", "mean"),
        avg_latency=("latency_seconds", "mean"),
        avg_memory_mb=("memory_mb", "mean"),
    ).reset_index()

    def norm(s):
        if s.max() == s.min():
            return pd.Series([0.5] * len(s), index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    # Judge-aware composite scoring
    has_judge = "judge_score" in df_valid.columns and df_valid["judge_score"].notna().any()
    if has_judge:
        judge_avg = df_valid.groupby("model")["judge_score"].mean()
        stats = stats.merge(judge_avg.rename("avg_judge"), left_on="model", right_index=True)
        stats["composite_score"] = (
            0.40 * norm(stats["avg_judge"])
            + 0.30 * norm(stats["avg_tokens_per_second"])
            + 0.20 * norm(stats["avg_bleu"])
            + 0.10 * (1 - norm(stats["avg_memory_mb"]))
        )
    else:
        stats["composite_score"] = (
            0.3 * norm(stats["avg_tokens_per_second"])
            + 0.4 * norm(stats["avg_keyword_accuracy"])
            + 0.2 * norm(stats["avg_bleu"])
            + 0.1 * (1 - norm(stats["avg_memory_mb"]))
        )

    stats = stats.sort_values("composite_score", ascending=False)
    return stats.round(4).to_dict(orient="records")


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource("prompts://all")
def resource_prompts() -> str:
    """All evaluation prompts in prompts.json."""
    return json.dumps(load_prompts(), indent=2)


@mcp.resource("results://latest")
def resource_latest_results() -> str:
    """Latest evaluation results JSON."""
    data = _load_latest_results()
    if data is None:
        return json.dumps({"error": "No results found. Run an evaluation first."})
    return json.dumps(data, indent=2)


@mcp.resource("results://leaderboard")
def resource_leaderboard() -> str:
    """Composite-scored leaderboard from the latest evaluation."""
    data = _load_latest_results()
    if data is None:
        return json.dumps({"error": "No results found. Run an evaluation first."})
    return json.dumps(_compute_leaderboard(data), indent=2)


# ---------------------------------------------------------------------------
# Tools — Prompt management (from Syed)
# ---------------------------------------------------------------------------

@mcp.tool()
def list_prompts() -> str:
    """List all current evaluation prompts with their IDs, categories, and difficulties."""
    prompts = load_prompts()
    summary = [
        {
            "id": p["id"],
            "category": p["category"],
            "difficulty": p.get("difficulty", "unknown"),
            "scoring_method": p.get("scoring_method", "keyword"),
            "prompt_preview": p["prompt"][:80] + ("…" if len(p["prompt"]) > 80 else ""),
            "tags": p.get("tags", []),
        }
        for p in prompts
    ]
    return json.dumps(summary, indent=2)


@mcp.tool()
def add_prompt(
    category: str,
    prompt: str,
    difficulty: str = "medium",
    scoring_method: str = "keyword",
    reference_answer: str = "",
    expected_keywords: list[str] | None = None,
    negated_keywords: list[str] | None = None,
    max_tokens: int = 200,
    tags: list[str] | None = None,
    rubric_hints: str = "",
    notes: str = "",
) -> str:
    """Add a new evaluation prompt to prompts.json."""
    if not prompt.strip():
        return "Error: prompt text cannot be empty."

    prompts = load_prompts()
    short_id = uuid.uuid4().hex[:6]
    new_prompt = {
        "id": f"{category.lower()}_{short_id}",
        "category": category,
        "difficulty": difficulty,
        "scoring_method": scoring_method,
        "prompt": prompt.strip(),
        "system_prompt": None,
        "reference_answer": reference_answer.strip(),
        "expected_keywords": expected_keywords or [],
        "negated_keywords": negated_keywords or [],
        "test_cases": [],
        "max_tokens": max_tokens,
        "tags": (tags or []) + ["mcp-generated"],
        "rubric_hints": rubric_hints.strip(),
        "notes": notes.strip(),
    }
    prompts.append(new_prompt)
    _save_prompts(prompts)
    return f"Added prompt '{new_prompt['id']}'. Total prompts: {len(prompts)}."


@mcp.tool()
def delete_prompt(prompt_id: str) -> str:
    """Remove a prompt from prompts.json by its ID."""
    prompts = load_prompts()
    original_count = len(prompts)
    remaining = [p for p in prompts if p["id"] != prompt_id]
    if len(remaining) == original_count:
        return f"Error: No prompt found with id '{prompt_id}'."
    _save_prompts(remaining)
    return f"Deleted '{prompt_id}'. Prompts remaining: {len(remaining)}."


@mcp.tool()
def get_prompt(prompt_id: str) -> str:
    """Get the full details of a single prompt by ID."""
    prompts = load_prompts()
    for p in prompts:
        if p["id"] == prompt_id:
            return json.dumps(p, indent=2)
    return f"Error: No prompt found with id '{prompt_id}'."


# ---------------------------------------------------------------------------
# Tools — Evaluation
# ---------------------------------------------------------------------------

@mcp.tool()
def run_evaluation(models: list[str] | None = None) -> str:
    """Trigger a full evaluation run against all (or specified) models."""
    if models:
        invalid = [m for m in models if m not in MODELS]
        if invalid:
            return f"Error: Unknown models: {invalid}. Valid: {list(MODELS.keys())}"
        available = models
    else:
        available = wait_for_models(timeout=60)
        if not available:
            return "Error: No models are reachable. Make sure containers are running."

    results = evaluate_all(available_models=available)
    path = save_results(results)
    return f"Evaluation complete. {len(results)} results saved to {os.path.basename(path)}."


@mcp.tool()
def get_results(category: str | None = None) -> str:
    """Return results from the latest evaluation run, optionally filtered by category."""
    data = _load_latest_results()
    if data is None:
        return "No results found. Run an evaluation first."
    if category:
        data = [r for r in data if r.get("category", "").lower() == category.lower()]
        if not data:
            return f"No results for category '{category}'."
    return json.dumps(data, indent=2)


@mcp.tool()
def compare_models(category: str | None = None) -> str:
    """Return the leaderboard with composite scores (judge-aware if available)."""
    data = _load_latest_results()
    if data is None:
        return "No results found. Run an evaluation first."
    if category:
        data = [r for r in data if r.get("category", "").lower() == category.lower()]
    leaderboard = _compute_leaderboard(data)
    if not leaderboard:
        return "No valid results to rank."
    return json.dumps(leaderboard, indent=2)


# ---------------------------------------------------------------------------
# Tools — Developer Pipeline (new direction)
# ---------------------------------------------------------------------------

@mcp.tool()
def analyze_repo(github_url: str) -> str:
    """Analyze a GitHub repository with Claude Sonnet to determine the best LLM task and generate tailored eval prompts."""
    try:
        analysis = analyze_github_url(github_url)
        return json.dumps(analysis, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def find_models(task: str, size: str = "medium", count: int = 3) -> str:
    """Search HuggingFace for suitable models matching a task type. Tasks: code-generation, text-generation, summarization, question-answering, translation, classification."""
    models = search_models(task, size, count)
    return json.dumps(models, indent=2)


@mcp.tool()
def setup_arena(github_url: str) -> str:
    """Full pipeline: analyze repo, find models, start containers, generate prompts, run evaluation."""
    try:
        # Step 1: Analyze repo
        analysis = analyze_github_url(github_url)
        task = analysis.get("llm_task", "text-generation")
        size = analysis.get("size_constraint", "medium")

        # Step 2: Find models
        models = search_models(task, size, count=3)
        model_pull_names = [m["pull_name"] for m in models]

        # Step 3: Write generated prompts
        suggested = analysis.get("suggested_prompts", [])
        if suggested:
            for p in suggested:
                p.setdefault("system_prompt", None)
                p.setdefault("negated_keywords", [])
                p.setdefault("test_cases", [])
                p.setdefault("tags", ["auto-generated"])
                p.setdefault("notes", p.get("rubric_hints", ""))
            _save_prompts(suggested)

        # Step 4: Start containers and pull models
        models_dict = setup_containers(model_pull_names)
        if not models_dict:
            return json.dumps({"error": "Failed to set up containers"})

        set_models(models_dict)

        # Step 5: Run evaluation
        results = evaluate_all(available_models=list(models_dict.keys()))
        path = save_results(results)

        return json.dumps({
            "status": "complete",
            "project_summary": analysis.get("project_summary", ""),
            "task": task,
            "models": list(models_dict.keys()),
            "prompts_count": len(suggested),
            "results_count": len(results),
            "results_file": os.path.basename(path),
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tools — Claude Judge (from Igor)
# ---------------------------------------------------------------------------

@mcp.tool()
def judge_response(category: str, prompt: str, response: str, reference_answer: str = "") -> str:
    """Use Claude as judge to evaluate a single LLM response with rubric-based scoring."""
    if _judge is None:
        return json.dumps({"error": "ANTHROPIC_API_KEY not set. Judge unavailable."})
    result = _judge.evaluate_response(category, prompt, response, reference_answer)
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
