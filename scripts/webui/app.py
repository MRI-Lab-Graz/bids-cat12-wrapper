#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, Response, jsonify, render_template, request, send_file
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
    WORKSPACE_ROOT / "projects" / "demo" / "project_config.json",
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
    report_path: Optional[str] = None  # set by generate_report_async on success

    def reset(self) -> None:
        with self.lock:
            self.lines = []
            self.exit_code = None
            self.report_path = None

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
                "report_path": self.report_path,
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


def resolve_browser_path(raw_path: str | None) -> Path:
    candidate = (raw_path or "").strip()
    if not candidate:
        return WORKSPACE_ROOT.resolve()

    target = Path(candidate)
    if target.is_absolute():
        return target.resolve()

    return (WORKSPACE_ROOT / target).resolve()


def to_browser_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        rel = str(resolved.relative_to(WORKSPACE_ROOT.resolve()))
        return "" if rel == "." else rel
    except Exception:
        return str(resolved)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def sidecar_path_for(project_path: Path) -> Path:
    suffix = project_path.suffix or ".json"
    return project_path.with_suffix(f"{suffix}.webui.json")


def resolve_project_base(config_data: Dict[str, Any]) -> Path:
    study_folder = (
        config_data.get("study", {})
        .get("project_folder")
    )
    if study_folder:
        p = Path(str(study_folder)).expanduser()
        if p.is_absolute():
            return p.resolve()
        return (WORKSPACE_ROOT / p).resolve()
    return WORKSPACE_ROOT


def resolve_from_config_path(config_data: Dict[str, Any], raw_value: str | None) -> Optional[Path]:
    if not raw_value:
        return None

    candidate = Path(str(raw_value)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    base = resolve_project_base(config_data)
    return (base / candidate).resolve()


def infer_run_options(config_data: Dict[str, Any]) -> Dict[str, Any]:
    stats_input = config_data.get("statistics", {}).get("input", {})
    return {
        "mode": "stats",
        "cat12_dir": stats_input.get("cat12_dir", ""),
        "participants": stats_input.get("participants_file", ""),
        "stats_config": "",
        "modality": "",
        "force_all": False,
        "dry_run": False,
    }


def build_stats_runtime_config(project_config: Dict[str, Any], runtime_config: Path) -> Path:
    software = project_config.get("software", {})
    statistics = project_config.get("statistics", {})
    stats_input = statistics.get("input", {})
    stats_design = statistics.get("design", {})
    stats_inference = statistics.get("inference", {})
    stats_exec = statistics.get("execution", {})

    stats_config: Dict[str, Any] = {
        "matlab": {
            "executable": software.get("matlab", {}).get("executable", "matlab"),
            "allow_graphics": bool(software.get("matlab", {}).get("allow_graphics", False)),
        },
        "spm": {
            "path": software.get("spm", {}).get("path", ""),
        },
        "analysis": {
            "participants_file": stats_input.get("participants_file", ""),
            "group_column": stats_input.get("group_column", "group"),
            "session_column": stats_input.get("session_column", "session"),
            "sessions": stats_input.get("sessions", ["all"]),
            "standardize_continuous": bool(stats_design.get("standardize_continuous", True)),
            "modalities": stats_design.get("modalities", []),
        },
        "screening": stats_inference.get("screening", {}),
        "tfce": stats_inference.get("tfce", {}),
        "reporting": statistics.get("reporting", {}),
        "performance": {
            "parallel_jobs": int(stats_exec.get("parallel_jobs", 1)),
            "memory_limit_gb": int(stats_exec.get("memory_limit_gb", 16)),
        },
        "output": {
            "analysis_name": stats_exec.get("analysis_name", "analysis"),
            "force_clean": bool(stats_exec.get("force_clean", False)),
        },
    }

    output_path = runtime_config.with_name("runtime_stats_config.json")
    save_json(output_path, stats_config)
    return output_path


def validate_config(config_path: Path) -> tuple[bool, List[str]]:
    try:
        cfg = load_json(config_path)
    except Exception as exc:
        return False, [f"Invalid JSON: {exc}"]

    is_unified = isinstance(cfg, dict) and all(k in cfg for k in ["study", "preprocessing", "statistics"]) 
    if is_unified:
        errors: List[str] = []

        stats_input = cfg.get("statistics", {}).get("input", {})
        stats_design = cfg.get("statistics", {}).get("design", {})
        preproc_bids = cfg.get("preprocessing", {}).get("bids", {})

        if not cfg.get("study", {}).get("project_folder"):
            errors.append("study.project_folder is required")
        if not preproc_bids.get("bids_dir"):
            errors.append("preprocessing.bids.bids_dir is required")
        if not stats_input.get("cat12_dir"):
            errors.append("statistics.input.cat12_dir is required")
        if not stats_input.get("participants_file"):
            errors.append("statistics.input.participants_file is required")

        modalities = stats_design.get("modalities", [])
        if not isinstance(modalities, list) or len(modalities) == 0:
            errors.append("statistics.design.modalities must be a non-empty array")

        return (len(errors) == 0), errors

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


def build_pipeline_command(runtime_config: Path, config_data: Dict[str, Any], run_options: Dict[str, Any]) -> List[str]:
    mode = str(run_options.get("mode", "stats") or "stats").strip().lower()
    dry_run = bool(run_options.get("dry_run"))

    preproc_cmd = [
        str(WORKSPACE_ROOT / "cat12_prepro"),
        "--config",
        str(runtime_config),
    ]
    if dry_run:
        preproc_cmd.append("--dry-run")

    stats_config_opt = run_options.get("stats_config")
    if stats_config_opt:
        stats_config_path = resolve_from_config_path(config_data, str(stats_config_opt))
        if not stats_config_path:
            raise ValueError("Invalid stats_config path")
    else:
        stats_config_path = build_stats_runtime_config(config_data, runtime_config)

    cat12_dir = run_options.get("cat12_dir") or config_data.get("statistics", {}).get("input", {}).get("cat12_dir")
    participants = run_options.get("participants") or config_data.get("statistics", {}).get("input", {}).get("participants_file")

    cat12_dir_path = resolve_from_config_path(config_data, str(cat12_dir) if cat12_dir else None)
    participants_path = resolve_from_config_path(config_data, str(participants) if participants else None)

    if not cat12_dir_path and mode in {"stats", "full"}:
        raise ValueError("CAT12 directory is required for statistics run")

    stats_cmd = [
        "bash",
        str(WORKSPACE_ROOT / "scripts" / "analysis" / "cat12_multi_modality.sh"),
        "--config",
        str(stats_config_path),
        "--cat12-dir",
        str(cat12_dir_path),
    ]

    if participants_path:
        stats_cmd.extend(["--participants", str(participants_path)])

    modality = str(run_options.get("modality", "")).strip()
    if modality:
        stats_cmd.extend(["--modality", modality])

    if bool(run_options.get("force_all")):
        stats_cmd.append("--force-all")

    if mode == "preproc":
        return preproc_cmd
    if mode == "stats":
        return stats_cmd
    if mode == "full":
        return ["bash", "-lc", f"{shlex.join(preproc_cmd)} && {shlex.join(stats_cmd)}"]

    raise ValueError("mode must be one of: preproc, stats, full")


def list_html_reports(config_data: Dict[str, Any], run_options: Dict[str, Any]) -> List[str]:
    candidates: set[str] = set()
    search_roots: List[Path] = []

    from_override = run_options.get("results_dir")
    if from_override:
        p = resolve_from_config_path(config_data, str(from_override))
        if p:
            search_roots.append(p)

    project_base = resolve_project_base(config_data)
    search_roots.append(project_base / "results")
    search_roots.append(project_base)
    search_roots.append(WORKSPACE_ROOT / "results")

    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for html in root.rglob("*.html"):
            if html.name.startswith("."):
                continue
            try:
                rel = str(html.resolve().relative_to(WORKSPACE_ROOT.resolve()))
            except Exception:
                rel = str(html.resolve())
            candidates.add(rel)

    return sorted(candidates, key=lambda r: Path(r).stat().st_mtime if Path(r).exists() else 0, reverse=True)


def default_report_results_dir(config_data: Dict[str, Any]) -> Optional[Path]:
    project_base = resolve_project_base(config_data)
    stats_output = config_data.get("statistics", {}).get("execution", {}).get("output_dir")
    if stats_output:
        p = resolve_from_config_path(config_data, str(stats_output))
        if p:
            return p

    fallback = project_base / "results"
    return fallback


def generate_report_async(command: List[str], output_html: Path) -> None:
    RUN_STATE.reset()
    RUN_STATE.append_line(f"$ {' '.join(shlex.quote(c) for c in command)}")

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
        if code == 0:
            RUN_STATE.report_path = str(output_html.resolve())

    RUN_STATE.append_line(f"[report generation finished] exit_code={code}")


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
        loaded = load_json(project_path)
        if isinstance(loaded, dict) and "config_data" in loaded:
            config_data = loaded.get("config_data", {})
            run_options = loaded.get("run_options", infer_run_options(config_data))
        else:
            config_data = loaded if isinstance(loaded, dict) else {}
            sidecar = sidecar_path_for(project_path)
            if sidecar.exists():
                side_data = load_json(sidecar)
                run_options = side_data.get("run_options", infer_run_options(config_data))
            else:
                run_options = infer_run_options(config_data)

        project_data = {
            "project_path": str(project_path),
            "config_path": str(project_path),
            "config_data": config_data,
            "run_options": run_options,
        }
    else:
        cfg_path = resolve_default_config()
        config_data = load_json(cfg_path)
        project_data = {
            "project_path": str(project_path),
            "config_path": str(cfg_path),
            "config_data": config_data,
            "run_options": infer_run_options(config_data),
        }

    return jsonify({"success": True, "project": project_data})


@app.route("/api/project/save", methods=["POST"])
def api_project_save() -> Response:
    payload = request.get_json(silent=True) or {}
    project_path = resolve_path(payload.get("project_path"), DEFAULT_PROJECT_PATH)

    config_data = payload.get("config_data", {})
    run_options = payload.get("run_options", {})

    save_json(project_path, config_data)
    save_json(
        sidecar_path_for(project_path),
        {
            "run_options": run_options,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
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

    current = resolve_browser_path(raw_path)
    if current.is_file():
        current = current.parent
    if not current.exists() or not current.is_dir():
        existing_parent = current.parent if current.parent.exists() and current.parent.is_dir() else WORKSPACE_ROOT.resolve()
        current = existing_parent

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

        entries.append({"name": name, "path": to_browser_path(entry), "is_dir": is_dir})

    current_rel = to_browser_path(current)

    parent_rel = ""
    if current.parent != current:
        parent_rel = to_browser_path(current.parent)

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

    try:
        command = build_pipeline_command(runtime_config, config_data, run_options)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

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


@app.route("/api/reports/list", methods=["POST"])
def api_reports_list() -> Response:
    payload = request.get_json(silent=True) or {}
    config_data = payload.get("config_data", {})
    run_options = payload.get("run_options", {})
    reports = list_html_reports(config_data, run_options)
    return jsonify({"success": True, "reports": reports})


@app.route("/api/reports/open", methods=["GET"])
def api_reports_open() -> Response:
    raw = request.args.get("path", "")
    path = resolve_within_workspace(raw)

    if not path.exists() or not path.is_file() or path.suffix.lower() != ".html":
        return jsonify({"success": False, "error": "Report not found"}), 404

    return send_file(path, mimetype="text/html")


@app.route("/api/reports/generate", methods=["POST"])
def api_reports_generate() -> Response:
    payload = request.get_json(silent=True) or {}
    config_data = payload.get("config_data", {})

    results_dir_raw = payload.get("results_dir")
    quality = str(payload.get("quality", "low") or "low")
    report_filter = str(payload.get("report_filter", "no_tfce") or "no_tfce")
    output_html_raw = payload.get("output_html")

    if results_dir_raw:
        results_dir = resolve_from_config_path(config_data, str(results_dir_raw))
    else:
        results_dir = default_report_results_dir(config_data)

    if not results_dir or not results_dir.exists() or not results_dir.is_dir():
        return jsonify({"success": False, "error": "results_dir is missing or does not exist"}), 400

    if output_html_raw:
        output_html = resolve_from_config_path(config_data, str(output_html_raw))
    else:
        output_html = results_dir / f"report_gui_{time.strftime('%Y%m%d_%H%M%S')}.html"

    if not output_html:
        return jsonify({"success": False, "error": "Invalid output_html path"}), 400

    # Prevent concurrent runs
    with RUN_STATE.lock:
        if RUN_STATE.running:
            return jsonify({"success": False, "error": "Another process is already running."}), 409

    runtime_cfg = WORKSPACE_ROOT / "scripts" / "webui" / ".runtime_report_config.json"
    save_json(runtime_cfg, config_data if isinstance(config_data, dict) else {})

    cmd = [
        sys.executable,
        str(WORKSPACE_ROOT / "scripts" / "reporting" / "post_stats_report.py"),
        str(results_dir),
        str(output_html),
        "--quality",
        quality,
        "--filter",
        report_filter,
        "--config",
        str(runtime_cfg),
    ]

    thread = threading.Thread(target=generate_report_async, args=(cmd, output_html), daemon=True)
    thread.start()

    return jsonify({"success": True, "message": "Report generation started"})


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
