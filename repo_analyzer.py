"""
Repo Analyzer — Fetches GitHub repo contents and uses Claude Sonnet to analyze
what LLM task would benefit the project, then generates tailored eval prompts.
"""

import json
import os
import re
from urllib.parse import urlparse

import requests
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# Map llm_task → which prompt categories are relevant for evaluation
TASK_TO_CATEGORIES = {
    "code-generation": ["Coding"],
    "text-generation": ["Coding", "Reasoning", "Creative"],
    "summarization": ["Summarization"],
    "question-answering": ["Factual", "QA", "Reasoning"],
    "translation": ["Creative"],
    "classification": ["Classification", "Reasoning", "Coding"],
}

SOURCE_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".go", ".rs", ".java", ".rb", ".php", ".c", ".cpp", ".h"}
MANIFEST_FILES = {"package.json", "requirements.txt", "Cargo.toml", "go.mod", "Gemfile", "composer.json", "pom.xml", "build.gradle"}
MAX_SOURCE_FILES = 8
MAX_FILE_SIZE = 8000  # characters per file


def parse_github_url(url: str) -> tuple[str, str]:
    """Extract (owner, repo) from a GitHub URL."""
    url = url.rstrip("/")
    # Handle: https://github.com/owner/repo or github.com/owner/repo
    parsed = urlparse(url if "://" in url else f"https://{url}")
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid GitHub URL: {url}")
    owner, repo = parts[0], parts[1].removesuffix(".git")
    return owner, repo


def _github_headers(token: str | None = None) -> dict:
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = token or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def fetch_repo_contents(github_url: str, token: str | None = None) -> dict:
    """Fetch README, manifest files, and key source files from a GitHub repo."""
    owner, repo = parse_github_url(github_url)
    headers = _github_headers(token)
    base = f"https://api.github.com/repos/{owner}/{repo}"

    result = {"owner": owner, "repo": repo, "readme": "", "files": {}, "languages": []}

    # Fetch README
    try:
        resp = requests.get(f"{base}/readme", headers=headers, timeout=15)
        if resp.ok:
            import base64
            data = resp.json()
            content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")
            result["readme"] = content[:10000]
    except Exception:
        pass

    # Fetch languages
    try:
        resp = requests.get(f"{base}/languages", headers=headers, timeout=10)
        if resp.ok:
            result["languages"] = list(resp.json().keys())
    except Exception:
        pass

    # Fetch repo tree (top-level + one level deep in common dirs)
    try:
        resp = requests.get(f"{base}/git/trees/HEAD?recursive=1", headers=headers, timeout=15)
        if resp.ok:
            tree = resp.json().get("tree", [])
            # Collect manifest files
            for item in tree:
                if item["type"] == "blob" and os.path.basename(item["path"]) in MANIFEST_FILES:
                    _fetch_file(base, item["path"], headers, result["files"])

            # Collect source files (prioritize shorter paths = more important files)
            source_candidates = [
                item for item in tree
                if item["type"] == "blob"
                and os.path.splitext(item["path"])[1] in SOURCE_EXTENSIONS
                and "test" not in item["path"].lower()
                and "vendor" not in item["path"].lower()
                and "node_modules" not in item["path"]
            ]
            source_candidates.sort(key=lambda x: (x["path"].count("/"), len(x["path"])))

            for item in source_candidates[:MAX_SOURCE_FILES]:
                _fetch_file(base, item["path"], headers, result["files"])
    except Exception:
        pass

    return result


def _fetch_file(base_url: str, path: str, headers: dict, files_dict: dict):
    """Fetch a single file's content from GitHub."""
    try:
        resp = requests.get(f"{base_url}/contents/{path}", headers=headers, timeout=10)
        if resp.ok:
            import base64
            data = resp.json()
            if data.get("encoding") == "base64" and data.get("content"):
                content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                files_dict[path] = content[:MAX_FILE_SIZE]
    except Exception:
        pass


ANALYSIS_SYSTEM_PROMPT = """You are an expert software architect analyzing a GitHub repository to recommend the best local LLM for the developer's use case.

Analyze the provided repository contents and return a JSON object with:

1. "project_summary": 1-2 sentence description of what this project does
2. "primary_language": The main programming language (lowercase)
3. "llm_task": One of: "code-generation", "text-generation", "summarization", "question-answering", "translation", "classification"
4. "task_rationale": 1 sentence explaining why this LLM task fits the project
5. "size_constraint": "small" (under 3B params), "medium" (3-8B params) — prefer "medium" for code tasks
6. "eval_categories": List of 2-3 categories from ["Coding", "Reasoning", "Factual", "Creative", "Summarization", "QA"]
7. "suggested_prompts": Array of 5-8 evaluation prompts tailored to this specific project. Each prompt must test how well an LLM could assist a developer working on THIS codebase.

Each prompt in suggested_prompts must have:
- "id": "<category_lowercase>_<number>" (e.g. "coding_1")
- "category": One of the eval_categories
- "difficulty": "easy", "medium", or "hard"
- "scoring_method": "keyword" (for verifiable answers), "coherence_only" (for creative/open-ended), or "exact" (for precise answers)
- "prompt": The actual prompt text — must be specific to the project's domain/stack
- "reference_answer": Expected answer (can be empty for coherence_only)
- "expected_keywords": List of keywords that should appear in a good response
- "rubric_hints": Brief guidance for the judge
- "max_tokens": 100-400 depending on complexity

Example: For a Flask REST API project, generate prompts like:
- "Write a Flask route that handles PUT requests for updating a user profile with email validation"
- "Explain how Flask's request context works and when you'd use g vs session"

Return ONLY valid JSON, no markdown fences."""


def analyze_repo(repo_contents: dict) -> dict:
    """Send repo contents to Claude Sonnet for analysis."""
    client = Anthropic()

    # Build context from repo contents
    context_parts = []
    if repo_contents.get("readme"):
        context_parts.append(f"## README\n{repo_contents['readme']}")
    if repo_contents.get("languages"):
        context_parts.append(f"## Languages\n{', '.join(repo_contents['languages'])}")
    for path, content in repo_contents.get("files", {}).items():
        context_parts.append(f"## File: {path}\n```\n{content}\n```")

    context = "\n\n".join(context_parts)

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4000,
        system=ANALYSIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Analyze this GitHub repository ({repo_contents['owner']}/{repo_contents['repo']}):\n\n{context}"}],
    )

    response_text = message.content[0].text

    # Parse JSON from response (handle potential markdown fences)
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    raise ValueError(f"Failed to parse JSON from Claude response: {response_text[:200]}")


def analyze_github_url(github_url: str) -> dict:
    """Full pipeline: fetch repo contents and analyze with Claude Sonnet."""
    print(f"  Fetching repo contents from {github_url}...")
    contents = fetch_repo_contents(github_url)
    file_count = len(contents.get("files", {}))
    print(f"  Fetched README + {file_count} source files. Languages: {contents.get('languages', [])}")

    print("  Analyzing with Claude Sonnet...")
    analysis = analyze_repo(contents)
    print(f"  Analysis complete: task={analysis.get('llm_task')}, prompts={len(analysis.get('suggested_prompts', []))}")
    return analysis
