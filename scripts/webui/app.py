#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, Response, jsonify, render_template, request
from waitress import serve


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
UTILS_DIR = WORKSPACE_ROOT / "scripts" / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

try:
    from validate_config_json import validate_config_file  # type: ignore
except Exception:
    validate_config_file = None


DEFAULT_CONFIG_CANDIDATES = [
    WORKSPACE_ROOT / "config" / "config_14_2_26.json",
    WORKSPACE_ROOT / "config" / "config.json",
]
DEFAULT_PROJECT_PATH = WORKSPACE_ROOT / "projects" / "webui" / "project.json"


@dataclass
class RunState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    process: Optional[subprocess.Popen[str]] = None
    running: bool = False
    lines: List[str] = field(default_factory=list)
    exit_code: Optional[int] = None

    def reset(self) -> None:
        with self.lock:
            self.lines = []
            self.exit_code = None

    def append_line(self, line: str) -> None:
        with self.lock:
            self.lines.append(line.rstrip("\n"))
            if len(self.lines) > 6000:
                self.lines = self.lines[-6000:]

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "running": self.running,
                "exit_code": self.exit_code,
                "line_count": len(self.lines),
            }


RUN_STATE = RunState()
app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)


def resolve_default_config() -> Path:
    for candidate in DEFAULT_CONFIG_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_CONFIG_CANDIDATES[-1]


def resolve_path(raw_path: str | None, default: Path) -> Path:
    if not raw_path:
        return default
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (WORKSPACE_ROOT / candidate).resolve()


def read_participants_columns(path: Path) -> List[str]:
    if not path.exists() or not path.is_file():
        return []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, [])

    cleaned = [column.strip() for column in header if column and column.strip()]
    return cleaned


def list_participant_files() -> List[str]:
    candidates: set[str] = set()
    patterns = [
        "results/data/*participants*.tsv",
        "results/data/*.tsv",
        "projects/**/*.tsv",
    ]

    for pattern in patterns:
        for path in WORKSPACE_ROOT.glob(pattern):
            if path.is_file():
                try:
                    rel = path.resolve().relative_to(WORKSPACE_ROOT)
                    candidates.add(str(rel))
                except Exception:
                    candidates.add(str(path.resolve()))

    return sorted(candidates)


def list_project_files() -> List[str]:
    candidates: set[str] = set()
    patterns = [
        "projects/**/*.json",
    ]

    for pattern in patterns:
        for path in WORKSPACE_ROOT.glob(pattern):
            if not path.is_file():
                continue
            try:
                rel = path.resolve().relative_to(WORKSPACE_ROOT)
                candidates.add(str(rel))
            except Exception:
                candidates.add(str(path.resolve()))

    return sorted(candidates)


def list_config_files() -> List[str]:
    candidates: set[str] = set()
    patterns = [
        "config/**/*.json",
        "projects/**/*config*.json",
        "projects/**/*stats*.json",
    ]

    for pattern in patterns:
        for path in WORKSPACE_ROOT.glob(pattern):
            if not path.is_file():
                continue
            try:
                rel = path.resolve().relative_to(WORKSPACE_ROOT)
                candidates.add(str(rel))
            except Exception:
                candidates.add(str(path.resolve()))

    return sorted(candidates)


def list_cat12_dirs() -> List[str]:
    candidates: set[str] = set()
    patterns = [
        "**/derivatives/cat12",
        "**/derivatives/cat12/*",
    ]

    for pattern in patterns:
        for path in WORKSPACE_ROOT.glob(pattern):
            if not path.is_dir():
                continue
            try:
                rel = path.resolve().relative_to(WORKSPACE_ROOT)
                candidates.add(str(rel))
            except Exception:
                candidates.add(str(path.resolve()))

    return sorted(candidates)


def list_results_dirs() -> List[str]:
    candidates: set[str] = set()
    patterns = [
        "results/*/*",
        "results/*",
    ]

    for pattern in patterns:
        for path in WORKSPACE_ROOT.glob(pattern):
            if not path.is_dir():
                continue
            try:
                rel = path.resolve().relative_to(WORKSPACE_ROOT)
                candidates.add(str(rel))
            except Exception:
                candidates.add(str(path.resolve()))

    return sorted(candidates)


def resolve_within_workspace(raw_path: str | None) -> Path:
    candidate = (raw_path or "").strip()
    base = WORKSPACE_ROOT.resolve()
    target = (base / candidate).resolve() if candidate else base

    try:
        target.relative_to(base)
    except Exception:
        return base

    return target


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def validate_config(config_path: Path) -> tuple[bool, List[str]]:
    if validate_config_file is None:
        return False, [
            "Config validator unavailable.",
            "Ensure scripts/utils/validate_config_json.py is importable.",
        ]

    ok, errors = validate_config_file(config_path)
    return ok, errors


def write_runtime_config(project_path: Path, config_data: Dict[str, Any]) -> Path:
    runtime_config = project_path.with_name("runtime_config.json")
    save_json(runtime_config, config_data)
    return runtime_config


def build_pipeline_command(runtime_config: Path, run_options: Dict[str, Any]) -> List[str]:
    cmd = [
        sys.executable,
        str(WORKSPACE_ROOT / "run_pipeline.py"),
        "--config",
        str(runtime_config),
    ]

    def opt(name: str, flag: str) -> None:
        value = run_options.get(name)
        if value:
            cmd.extend([flag, str(value)])

    opt("cat12_dir", "--cat12-dir")
    opt("participants", "--participants")
    opt("results_dir", "--results-dir")
    opt("modality", "--modality")
    opt("only", "--only")
    opt("skip", "--skip")
    opt("from_step", "--from-step")
    opt("until_step", "--until-step")

    if run_options.get("use_matlab"):
        cmd.append("--use-matlab")
    if run_options.get("force"):
        cmd.append("--force")
    if run_options.get("dry_run"):
        cmd.append("--dry-run")

    return cmd


def run_pipeline_async(command: List[str]) -> None:
    RUN_STATE.reset()
    RUN_STATE.append_line(f"$ {' '.join(command)}")

    with RUN_STATE.lock:
        RUN_STATE.running = True
        RUN_STATE.process = subprocess.Popen(
            command,
            cwd=WORKSPACE_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    assert RUN_STATE.process is not None
    process = RUN_STATE.process

    for line in process.stdout or []:
        RUN_STATE.append_line(line)

    process.wait()
    code = process.returncode

    with RUN_STATE.lock:
        RUN_STATE.running = False
        RUN_STATE.exit_code = code
        RUN_STATE.process = None

    RUN_STATE.append_line(f"[pipeline finished] exit_code={code}")


@app.route("/")
def index() -> str:
    return render_template(
        "pages/index.html",
        default_config=str(resolve_default_config().relative_to(WORKSPACE_ROOT)),
        default_project=str(DEFAULT_PROJECT_PATH.relative_to(WORKSPACE_ROOT)),
    )


@app.route("/api/project/load", methods=["POST"])
def api_project_load() -> Response:
    payload = request.get_json(silent=True) or {}
    project_path = resolve_path(payload.get("project_path"), DEFAULT_PROJECT_PATH)

    if project_path.exists():
        project_data = load_json(project_path)
    else:
        cfg_path = resolve_default_config()
        config_data = load_json(cfg_path)
        default_participants = config_data.get("analysis", {}).get("participants_file", "")
        project_data = {
            "project_path": str(project_path),
            "config_path": str(cfg_path),
            "config_data": config_data,
            "run_options": {
                "only": "stats,report",
                "skip": "",
                "from_step": "",
                "until_step": "",
                "cat12_dir": "",
                "participants": default_participants,
                "results_dir": "",
                "modality": "",
                "use_matlab": False,
                "force": False,
                "dry_run": False,
            },
        }

    return jsonify({"success": True, "project": project_data})


@app.route("/api/project/save", methods=["POST"])
def api_project_save() -> Response:
    payload = request.get_json(silent=True) or {}
    project_path = resolve_path(payload.get("project_path"), DEFAULT_PROJECT_PATH)

    project_data = {
        "project_path": str(project_path),
        "config_path": payload.get("config_path") or str(resolve_default_config()),
        "config_data": payload.get("config_data", {}),
        "run_options": payload.get("run_options", {}),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    save_json(project_path, project_data)
    return jsonify({"success": True, "project_path": str(project_path)})


@app.route("/api/config/validate", methods=["POST"])
def api_config_validate() -> Response:
    payload = request.get_json(silent=True) or {}
    project_path = resolve_path(payload.get("project_path"), DEFAULT_PROJECT_PATH)
    config_data = payload.get("config_data", {})

    runtime_config = write_runtime_config(project_path, config_data)
    ok, errors = validate_config(runtime_config)
    return jsonify({"success": ok, "errors": errors})


@app.route("/api/participants/columns", methods=["POST"])
def api_participants_columns() -> Response:
    payload = request.get_json(silent=True) or {}
    raw_path = payload.get("participants_path")

    if not raw_path:
        return jsonify({"success": True, "columns": []})

    path = resolve_path(raw_path, WORKSPACE_ROOT)
    columns = read_participants_columns(path)
    return jsonify({"success": True, "columns": columns, "resolved_path": str(path)})


@app.route("/api/participants/files", methods=["GET"])
def api_participants_files() -> Response:
    files = list_participant_files()
    return jsonify({"success": True, "files": files})


@app.route("/api/project/files", methods=["GET"])
def api_project_files() -> Response:
    files = list_project_files()
    return jsonify({"success": True, "files": files})


@app.route("/api/config/files", methods=["GET"])
def api_config_files() -> Response:
    files = list_config_files()
    return jsonify({"success": True, "files": files})


@app.route("/api/cat12/dirs", methods=["GET"])
def api_cat12_dirs() -> Response:
    directories = list_cat12_dirs()
    return jsonify({"success": True, "dirs": directories})


@app.route("/api/results/dirs", methods=["GET"])
def api_results_dirs() -> Response:
    directories = list_results_dirs()
    return jsonify({"success": True, "dirs": directories})


@app.route("/api/fs/list", methods=["GET"])
def api_fs_list() -> Response:
    raw_path = request.args.get("path", "")
    allow_files = (request.args.get("files", "1") == "1")
    allow_dirs = (request.args.get("dirs", "1") == "1")
    exts_raw = request.args.get("ext", "")

    exts = [part.strip().lower() for part in exts_raw.split(",") if part.strip()]
    if exts:
        exts = [ext if ext.startswith(".") else f".{ext}" for ext in exts]

    current = resolve_within_workspace(raw_path)
    if current.is_file():
        current = current.parent
    if not current.exists() or not current.is_dir():
        current = WORKSPACE_ROOT.resolve()

    entries: List[Dict[str, Any]] = []
    for entry in sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        name = entry.name
        if name.startswith("."):
            continue

        is_dir = entry.is_dir()
        if is_dir and not allow_dirs:
            continue
        if (not is_dir) and not allow_files:
            continue
        if (not is_dir) and exts and entry.suffix.lower() not in exts:
            if not any(name.lower().endswith(ext) for ext in exts):
                continue

        try:
            rel = str(entry.resolve().relative_to(WORKSPACE_ROOT.resolve()))
        except Exception:
            continue

        entries.append({"name": name, "path": rel, "is_dir": is_dir})

    current_rel = str(current.relative_to(WORKSPACE_ROOT.resolve()))
    if current_rel == ".":
        current_rel = ""

    parent_rel = ""
    if current != WORKSPACE_ROOT.resolve():
        try:
            parent_rel = str(current.parent.resolve().relative_to(WORKSPACE_ROOT.resolve()))
            if parent_rel == ".":
                parent_rel = ""
        except Exception:
            parent_rel = ""

    return jsonify(
        {
            "success": True,
            "current": current_rel,
            "parent": parent_rel,
            "entries": entries,
        }
    )


@app.route("/api/run/start", methods=["POST"])
def api_run_start() -> Response:
    payload = request.get_json(silent=True) or {}
    project_path = resolve_path(payload.get("project_path"), DEFAULT_PROJECT_PATH)
    config_data = payload.get("config_data", {})
    run_options = payload.get("run_options", {})

    with RUN_STATE.lock:
        if RUN_STATE.running:
            return jsonify({"success": False, "error": "A pipeline is already running."}), 409

    runtime_config = write_runtime_config(project_path, config_data)
    ok, errors = validate_config(runtime_config)
    if not ok:
        return jsonify({"success": False, "error": "Config validation failed", "errors": errors}), 400

    project_data = {
        "project_path": str(project_path),
        "config_path": str(runtime_config),
        "config_data": config_data,
        "run_options": run_options,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(project_path, project_data)

    command = build_pipeline_command(runtime_config, run_options)
    thread = threading.Thread(target=run_pipeline_async, args=(command,), daemon=True)
    thread.start()

    return jsonify({"success": True, "command": command})


@app.route("/api/run/stop", methods=["POST"])
def api_run_stop() -> Response:
    with RUN_STATE.lock:
        process = RUN_STATE.process
        running = RUN_STATE.running

    if not running or process is None:
        return jsonify({"success": False, "error": "No running process."}), 400

    process.terminate()
    RUN_STATE.append_line("[requested stop] terminate signal sent")
    return jsonify({"success": True})


@app.route("/api/run/status", methods=["GET"])
def api_run_status() -> Response:
    return jsonify({"success": True, "status": RUN_STATE.snapshot()})


@app.route("/api/run/stream", methods=["GET"])
def api_run_stream() -> Response:
    def event_stream() -> Any:
        last_index = 0
        while True:
            with RUN_STATE.lock:
                lines = RUN_STATE.lines[last_index:]
                running = RUN_STATE.running
                exit_code = RUN_STATE.exit_code
                total = len(RUN_STATE.lines)

            for line in lines:
                payload = json.dumps({"type": "line", "text": line})
                yield f"data: {payload}\n\n"
                last_index += 1

            if not running and last_index >= total:
                payload = json.dumps({"type": "done", "exit_code": exit_code})
                yield f"data: {payload}\n\n"
                break

            time.sleep(0.3)

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/shutdown", methods=["POST"])
def shutdown() -> Response:
    os._exit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAT12 Web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def open_browser(host: str, port: int) -> None:
    url = f"http://{host}:{port}"
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()


def main() -> int:
    args = parse_args()

    if not args.no_browser:
        open_browser(args.host, args.port)

    print(f"Starting CAT12 Web UI at http://{args.host}:{args.port}")
    serve(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
