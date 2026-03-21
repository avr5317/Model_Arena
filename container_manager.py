"""
Container Manager — Dynamic Docker/Ollama container orchestration.
Replaces static docker-compose.yml with programmatic container lifecycle.
"""

import subprocess
import time

BASE_PORT = 11434
CONTAINER_PREFIX = "arena-model"
DEFAULT_MEMORY = "8g"
DEFAULT_CPUS = "4"


def start_container(name: str, port: int, memory_limit: str = DEFAULT_MEMORY) -> bool:
    """Start a new Ollama container with the given name and port."""
    try:
        # Remove existing container with same name if it exists
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True, timeout=10
        )

        volume_name = f"{name}-data"
        result = subprocess.run(
            [
                "docker", "run", "-d",
                "--name", name,
                "-p", f"{port}:11434",
                "--memory", memory_limit,
                "--cpus", DEFAULT_CPUS,
                "-v", f"{volume_name}:/root/.ollama",
                "ollama/ollama:latest",
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print(f"  [OK] Started container {name} on port {port}")
            return True
        else:
            print(f"  [ERR] Failed to start {name}: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"  [ERR] Failed to start {name}: {e}")
        return False


def wait_for_container_ready(name: str, port: int, timeout: int = 60) -> bool:
    """Wait until the Ollama API is responsive in the container."""
    import requests as req

    start = time.time()
    while (time.time() - start) < timeout:
        try:
            resp = req.get(f"http://localhost:{port}/api/tags", timeout=3)
            if resp.ok:
                return True
        except Exception:
            pass
        time.sleep(2)
    print(f"  [WARN] Container {name} not ready after {timeout}s")
    return False


def pull_model_in_container(container_name: str, model_pull_name: str, progress_callback=None) -> bool:
    """Pull a model inside an Ollama container. Streams progress."""
    try:
        print(f"  Pulling {model_pull_name} in {container_name}...")
        process = subprocess.Popen(
            ["docker", "exec", container_name, "ollama", "pull", model_pull_name],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

        for line in process.stdout:
            line = line.strip()
            if line and progress_callback:
                progress_callback(container_name, model_pull_name, line)
            elif line:
                print(f"    [{container_name}] {line}")

        process.wait(timeout=600)
        if process.returncode == 0:
            print(f"  [OK] Pulled {model_pull_name} in {container_name}")
            return True
        else:
            print(f"  [ERR] Failed to pull {model_pull_name} in {container_name}")
            return False
    except Exception as e:
        print(f"  [ERR] Pull failed for {model_pull_name}: {e}")
        return False


def stop_container(name: str) -> bool:
    """Stop and remove a container."""
    try:
        result = subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


def cleanup_all_arena_containers() -> None:
    """Stop and remove all arena-model-* containers and any containers using arena ports."""
    # Remove arena-model-* containers
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={CONTAINER_PREFIX}", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            containers = [c.strip() for c in result.stdout.strip().split("\n") if c.strip()]
            for name in containers:
                stop_container(name)
                print(f"  Removed {name}")
    except Exception:
        pass

    # Also stop any containers using our target ports (e.g. old ollama-phi3, ollama-llama3, ollama-mistral)
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Ports}}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split("\t")
                name = parts[0]
                ports = parts[1] if len(parts) > 1 else ""
                # Check if this container uses any of our target ports
                for i in range(3):
                    if f":{BASE_PORT + i}->" in ports:
                        print(f"  Stopping {name} (using port {BASE_PORT + i})")
                        stop_container(name)
                        break
    except Exception:
        pass


def get_running_containers() -> list[dict]:
    """Get info about running arena containers."""
    containers = []
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={CONTAINER_PREFIX}",
             "--format", "{{.Names}}\t{{.Ports}}\t{{.Status}}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split("\t")
                containers.append({
                    "name": parts[0],
                    "ports": parts[1] if len(parts) > 1 else "",
                    "status": parts[2] if len(parts) > 2 else "",
                })
    except Exception:
        pass
    return containers


def setup_containers(model_names: list[str], progress_callback=None) -> dict:
    """
    Full setup: start containers, wait for readiness, pull models.
    Returns a MODELS dict compatible with evaluate.py.
    """
    import concurrent.futures

    cleanup_all_arena_containers()

    n = len(model_names)
    containers = []

    # Start containers in parallel
    for i in range(n):
        name = f"{CONTAINER_PREFIX}-{i}"
        port = BASE_PORT + i
        if start_container(name, port):
            containers.append({"name": name, "port": port, "model": model_names[i]})

    if not containers:
        return {}

    # Wait for all containers to be ready
    print("  Waiting for containers to be ready...")
    for c in containers:
        wait_for_container_ready(c["name"], c["port"], timeout=60)

    # Pull models sequentially (parallel pulls can be interrupted by Streamlit reruns)
    print("  Pulling models (this may take several minutes)...")
    for c in containers:
        success = pull_model_in_container(c["name"], c["model"], progress_callback)
        if not success:
            print(f"  [WARN] Failed to pull model for {c['name']}")

    # Build MODELS dict for evaluate.py
    models_dict = {}
    for c in containers:
        models_dict[c["model"]] = {
            "url": f"http://localhost:{c['port']}",
            "container": c["name"],
        }

    return models_dict


def build_models_dict(model_names: list[str], containers: list[dict]) -> dict:
    """Build the MODELS dict that evaluate.py expects."""
    models_dict = {}
    for model_name, container in zip(model_names, containers):
        port = container.get("port", BASE_PORT)
        models_dict[model_name] = {
            "url": f"http://localhost:{port}",
            "container": container["name"],
        }
    return models_dict
