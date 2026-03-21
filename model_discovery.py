"""
Model Discovery — Searches HuggingFace for suitable models and provides
curated fallbacks for reliability.
"""

from huggingface_hub import HfApi

# Curated fallback models per task — known to work with Ollama
FALLBACK_MODELS = {
    "code-generation": [
        {"name": "qwen2.5-coder:3b", "pull_name": "qwen2.5-coder:3b", "params": "3B", "source": "ollama_library"},
        {"name": "codellama:7b", "pull_name": "codellama:7b", "params": "7B", "source": "ollama_library"},
        {"name": "deepseek-coder:6.7b", "pull_name": "deepseek-coder:6.7b", "params": "6.7B", "source": "ollama_library"},
    ],
    "text-generation": [
        {"name": "llama3.2:3b", "pull_name": "llama3.2:3b", "params": "3B", "source": "ollama_library"},
        {"name": "mistral:7b", "pull_name": "mistral:7b", "params": "7B", "source": "ollama_library"},
        {"name": "phi3:mini", "pull_name": "phi3:mini", "params": "3.8B", "source": "ollama_library"},
    ],
    "summarization": [
        {"name": "llama3.2:3b", "pull_name": "llama3.2:3b", "params": "3B", "source": "ollama_library"},
        {"name": "mistral:7b", "pull_name": "mistral:7b", "params": "7B", "source": "ollama_library"},
        {"name": "qwen2.5:3b", "pull_name": "qwen2.5:3b", "params": "3B", "source": "ollama_library"},
    ],
    "question-answering": [
        {"name": "llama3.2:3b", "pull_name": "llama3.2:3b", "params": "3B", "source": "ollama_library"},
        {"name": "phi3:mini", "pull_name": "phi3:mini", "params": "3.8B", "source": "ollama_library"},
        {"name": "mistral:7b", "pull_name": "mistral:7b", "params": "7B", "source": "ollama_library"},
    ],
    "translation": [
        {"name": "llama3.2:3b", "pull_name": "llama3.2:3b", "params": "3B", "source": "ollama_library"},
        {"name": "mistral:7b", "pull_name": "mistral:7b", "params": "7B", "source": "ollama_library"},
        {"name": "qwen2.5:3b", "pull_name": "qwen2.5:3b", "params": "3B", "source": "ollama_library"},
    ],
    "classification": [
        {"name": "phi3:mini", "pull_name": "phi3:mini", "params": "3.8B", "source": "ollama_library"},
        {"name": "llama3.2:3b", "pull_name": "llama3.2:3b", "params": "3B", "source": "ollama_library"},
        {"name": "mistral:7b", "pull_name": "mistral:7b", "params": "7B", "source": "ollama_library"},
    ],
}

# Map our task names to HuggingFace search parameters
TASK_TO_HF = {
    "code-generation": {"search": "code", "task": "text-generation"},
    "text-generation": {"search": "", "task": "text-generation"},
    "summarization": {"search": "summarize", "task": "text-generation"},
    "question-answering": {"search": "qa", "task": "text-generation"},
    "translation": {"search": "translation", "task": "text-generation"},
    "classification": {"search": "classify", "task": "text-generation"},
}

# Size limits in billions of parameters
SIZE_LIMITS = {
    "small": 3,
    "medium": 8,
    "large": 14,
}


def search_huggingface(task: str, size_constraint: str = "medium", count: int = 3) -> list[dict]:
    """Search HuggingFace for GGUF-compatible models matching the task."""
    hf_params = TASK_TO_HF.get(task, TASK_TO_HF["text-generation"])
    max_params_b = SIZE_LIMITS.get(size_constraint, 8)

    try:
        api = HfApi()
        models = api.list_models(
            search=hf_params.get("search", ""),
            pipeline_tag=hf_params.get("task", "text-generation"),
            filter="gguf",
            sort="downloads",
            limit=50,
        )

        candidates = []
        for model in models:
            model_id = model.id if hasattr(model, "id") else str(model)

            # Size filtering (rough heuristic from model name)
            param_size = _extract_param_size(model_id)
            if param_size and param_size > max_params_b:
                continue

            downloads = model.downloads if hasattr(model, "downloads") else 0
            candidates.append({
                "name": model_id.split("/")[-1],
                "pull_name": f"hf.co/{model_id}",
                "params": f"{param_size}B" if param_size else "unknown",
                "downloads": downloads,
                "source": "huggingface",
            })

            if len(candidates) >= count:
                break

        return candidates
    except Exception as e:
        print(f"  [WARN] HuggingFace search failed: {e}")
        return []


def _extract_param_size(model_name: str) -> float | None:
    """Try to extract parameter count (in billions) from model name."""
    import re
    # Match patterns like "3b", "7B", "6.7b", "3B"
    match = re.search(r"(\d+\.?\d*)[bB]", model_name)
    if match:
        return float(match.group(1))
    return None


def search_models(task: str, size_constraint: str = "medium", count: int = 3) -> list[dict]:
    """
    Find suitable models for the given task. Tries HuggingFace first,
    falls back to curated list if needed.
    """
    # Try HuggingFace search first
    hf_models = search_huggingface(task, size_constraint, count)

    if len(hf_models) >= count:
        print(f"  Found {len(hf_models)} models from HuggingFace")
        return hf_models[:count]

    # Fill remaining slots with fallback models
    fallback = FALLBACK_MODELS.get(task, FALLBACK_MODELS["text-generation"])
    result = list(hf_models)

    # Avoid duplicates
    existing_names = {m["pull_name"] for m in result}
    for fb in fallback:
        if fb["pull_name"] not in existing_names and len(result) < count:
            result.append(fb)
            existing_names.add(fb["pull_name"])

    if not hf_models:
        print(f"  Using fallback models for task '{task}'")
    else:
        print(f"  Found {len(hf_models)} from HuggingFace, filled {count - len(hf_models)} from fallbacks")

    return result[:count]
