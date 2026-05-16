# Copyright (c) 2026 Samrat Kar
# Licensed under CC BY-NC-SA 4.0 — see LICENSE for details.

"""System health checks for the RAG Research Assistant.

Verifies environment variables, required dependencies, and the knowledge-base
directory so operators can confirm the system is ready before users hit it.

Usage::

    python -m src.health          # human-readable report
    python -m src.health --json   # machine-readable JSON
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


_REQUIRED_PACKAGES = [
    "crewai",
    "langchain_openai",
    "streamlit",
    "numpy",
    "dotenv",
    "pydantic",
    "tiktoken",
]

_SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json"}


def check_environment() -> Dict[str, str]:
    """Return the status of required and optional env vars."""
    openai_key = os.getenv("OPENAI_API_KEY", "")
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    return {
        "OPENAI_API_KEY": "ok" if openai_key else "missing — LLM and embeddings will fail",
        "TAVILY_API_KEY": (
            "ok"
            if tavily_key
            else "missing — Full Research mode (web search) will fail; Quick Mode is unaffected"
        ),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini (default)"),
    }


def check_dependencies() -> Dict[str, str]:
    """Return the import status of each required package."""
    results: Dict[str, str] = {}
    for pkg in _REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            results[pkg] = "ok"
        except ImportError as exc:
            results[pkg] = f"missing — {exc}"
    return results


def check_data_directory(data_dir: str = "data") -> Dict[str, Any]:
    """Return the status of the knowledge-base directory."""
    p = Path(data_dir)
    if not p.exists():
        return {
            "status": "missing",
            "path": str(p.resolve()),
            "file_count": 0,
            "files": [],
            "message": f"Directory '{data_dir}' does not exist. Create it and add .txt/.md files.",
        }

    files = sorted(
        f.name for f in p.iterdir() if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTENSIONS
    )
    if not files:
        return {
            "status": "empty",
            "path": str(p.resolve()),
            "file_count": 0,
            "files": [],
            "message": f"Directory '{data_dir}' exists but contains no supported files (.txt, .md, .csv, .json).",
        }

    return {
        "status": "ok",
        "path": str(p.resolve()),
        "file_count": len(files),
        "files": files,
        "message": f"{len(files)} document(s) ready for ingestion.",
    }


def check_outputs_directory() -> Dict[str, Any]:
    """Return the write-access status of the outputs directory."""
    out = Path("outputs")
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".health_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return {"status": "ok", "path": str(out.resolve()), "writable": True}
    except OSError as exc:
        return {
            "status": "error",
            "path": str(out.resolve()),
            "writable": False,
            "message": str(exc),
        }


def run_health_check(data_dir: str = "data") -> Dict[str, Any]:
    """Run all checks and return a structured report."""
    env = check_environment()
    deps = check_dependencies()
    data = check_data_directory(data_dir)
    outputs = check_outputs_directory()

    env_ok = env["OPENAI_API_KEY"] == "ok"
    deps_ok = all(v == "ok" for v in deps.values())
    data_ok = data["status"] == "ok"
    outputs_ok = outputs["status"] == "ok"

    overall = "healthy" if (env_ok and deps_ok and data_ok and outputs_ok) else "degraded"

    return {
        "status": overall,
        "environment": env,
        "dependencies": deps,
        "data_directory": data,
        "outputs_directory": outputs,
    }


def _icon(ok: bool) -> str:
    return "OK  " if ok else "FAIL"


def print_health_report(data_dir: str = "data") -> int:
    """Print a human-readable health report. Returns 0 if healthy, 1 if degraded."""
    report = run_health_check(data_dir)

    print(f"\n=== Health Check: {report['status'].upper()} ===\n")

    print("Environment Variables:")
    for key, val in report["environment"].items():
        ok = val == "ok" or "default" in val
        print(f"  [{_icon(ok)}] {key}: {val}")

    print("\nDependencies:")
    for pkg, val in report["dependencies"].items():
        ok = val == "ok"
        print(f"  [{_icon(ok)}] {pkg}: {val}")

    print("\nData Directory:")
    d = report["data_directory"]
    ok = d["status"] == "ok"
    print(f"  [{_icon(ok)}] Status: {d['status']} — {d.get('message', '')}")
    if d["files"]:
        for name in d["files"]:
            print(f"             - {name}")

    print("\nOutputs Directory:")
    o = report["outputs_directory"]
    ok = o["status"] == "ok"
    print(f"  [{_icon(ok)}] Status: {o['status']} — {o['path']}")

    print()
    return 0 if report["status"] == "healthy" else 1


if __name__ == "__main__":
    if "--json" in sys.argv:
        data_dir = "data"
        for arg in sys.argv[1:]:
            if not arg.startswith("-"):
                data_dir = arg
        print(json.dumps(run_health_check(data_dir), indent=2))
        sys.exit(0)

    data_dir = "data"
    for arg in sys.argv[1:]:
        if not arg.startswith("-"):
            data_dir = arg
    sys.exit(print_health_report(data_dir))
