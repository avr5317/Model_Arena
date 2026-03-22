"""Keep result JSON lists compatible; store metadata in a separate directory."""
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path


class EvaluationResults(list):
    def __init__(self, rows=(), metadata=None):
        super().__init__(rows)
        self.metadata = metadata or {}


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_metadata(prompts, models, judge_model, project_root):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root,
            capture_output=True, text=True, timeout=3, check=True,
        )
        commit = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "prompts": prompts,
        "prompt_set_hash": fingerprint(prompts),
        "models": list(models),
        "judge_model": judge_model,
        "evaluator_commit": commit,
        "host": {"system": platform.system(), "machine": platform.machine(),
                 "python": platform.python_version(), "cpu_count": os.cpu_count()},
        "generation": {"stream": False, "num_predict": "per prompt; default 200",
                       "other_options": "Ollama defaults; not pinned"},
        "limitations": ["Model tags are recorded, not immutable model digests.",
                        "Host details do not capture GPU, RAM, or Docker allocation.",
                        "Evaluator commit is not the analyzed repository commit."],
    }


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def save_run(results, directory, run_name=None, repository_url=None):
    directory = Path(directory)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"eval_{stamp}_{uuid.uuid4().hex[:8]}"
    path = directory / f"{run_id}.json"
    metadata = dict(getattr(results, "metadata", {}))
    metadata.update({"run_id": run_id, "name": (run_name or "").strip() or run_id,
                     "saved_at": datetime.now(timezone.utc).isoformat(),
                     "repository_url": repository_url})
    # Publishing the result last prevents the UI from seeing a half-saved run.
    _atomic_json(directory / "metadata" / path.name, metadata)
    _atomic_json(path, list(results))
    return str(path)


def load_run(path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("model"), str)
        or not isinstance(row.get("prompt_id"), (str, int)) for row in rows
    ):
        raise ValueError("Expected a list of results with model and prompt_id fields.")
    metadata_path = path.parent / "metadata" / path.name
    warnings = []
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("Metadata must be an object")
        except (OSError, ValueError):
            metadata = {}
            warnings.append("Metadata could not be read; treating this as a legacy run.")
    return {"rows": rows, "metadata": metadata, "path": str(path),
            "name": str(metadata.get("name") or path.stem), "warnings": warnings}
