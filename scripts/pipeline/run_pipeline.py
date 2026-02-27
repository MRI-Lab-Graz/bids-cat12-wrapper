#!/usr/bin/env python3
"""
Unified CAT12 pipeline runner (draft).

Goal:
- One command with one config JSON
- No flags: run all possible steps
- With flags: select/skip/range of steps

Current step chain:
  preproc -> stats -> sweep -> report

Notes:
- This draft wraps existing scripts and keeps current behavior intact.
- `preproc` is optional and only runs if configured via `pipeline.preproc_command`.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


STEP_ORDER = ["preproc", "stats", "sweep", "report"]


@dataclass
class RunContext:
    workspace_root: Path
    config_path: Path
    config: Dict
    cat12_dir: Optional[Path]
    participants_file: Optional[Path]
    modality: Optional[str]
    results_dir_override: Optional[Path]
    use_matlab: bool
    force: bool
    dry_run: bool


def _parse_step_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _validate_steps(steps: Sequence[str]) -> None:
    bad = [step for step in steps if step not in STEP_ORDER]
    if bad:
        raise ValueError(
            f"Unknown step(s): {', '.join(bad)}. Valid: {', '.join(STEP_ORDER)}"
        )


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _enabled_by_config(config: Dict, step: str) -> bool:
    pipeline_cfg = config.get("pipeline", {})
    step_cfg = pipeline_cfg.get("steps", {}).get(step, {})
    if "enabled" in step_cfg:
        return bool(step_cfg["enabled"])

    if step == "preproc":
        return bool(pipeline_cfg.get("preproc_command"))
    if step == "report":
        return bool(config.get("reporting", {}).get("auto_generate", True))
    if step == "sweep":
        return bool(
            config.get("double_threshold", {}).get("enabled", True)
            or config.get("tfce", {}).get("enabled", False)
            or config.get("reporting", {}).get("enabled", True)
        )

    return True


def _select_steps(
    config: Dict,
    only: List[str],
    skip: List[str],
    from_step: Optional[str],
    until_step: Optional[str],
) -> List[str]:
    _validate_steps(only)
    _validate_steps(skip)
    _validate_steps([from_step] if from_step else [])
    _validate_steps([until_step] if until_step else [])

    if only:
        selected = [step for step in STEP_ORDER if step in only]
    else:
        selected = [step for step in STEP_ORDER if _enabled_by_config(config, step)]

    if from_step:
        start = STEP_ORDER.index(from_step)
        selected = [step for step in selected if STEP_ORDER.index(step) >= start]

    if until_step:
        end = STEP_ORDER.index(until_step)
        selected = [step for step in selected if STEP_ORDER.index(step) <= end]

    if skip:
        skip_set = set(skip)
        selected = [step for step in selected if step not in skip_set]

    return selected


def _run_command(cmd: List[str], ctx: RunContext, label: str) -> int:
    printable = shlex.join(cmd)
    print(f"\n[{label}] {printable}")

    if ctx.dry_run:
        return 0

    result = subprocess.run(cmd, cwd=ctx.workspace_root)
    return result.returncode


def _resolve_cat12_dir(ctx: RunContext) -> Optional[Path]:
    if ctx.cat12_dir:
        return ctx.cat12_dir

    from_config = (
        ctx.config.get("analysis", {}).get("cat12_dir")
        or ctx.config.get("paths", {}).get("cat12_dir")
        or ctx.config.get("study", {}).get("cat12_dir")
    )
    if from_config:
        return Path(from_config)

    return None


def _resolve_participants(ctx: RunContext) -> Optional[Path]:
    if ctx.participants_file:
        return ctx.participants_file

    from_config = ctx.config.get("analysis", {}).get("participants_file")
    return Path(from_config) if from_config else None


def _modalities_from_config(config: Dict) -> List[Dict]:
    return config.get("analysis", {}).get("modalities", [])


def _resolve_results_dirs(ctx: RunContext) -> List[Path]:
    if ctx.results_dir_override:
        return [ctx.results_dir_override]

    modalities = _modalities_from_config(ctx.config)
    dirs: List[Path] = []
    for modality in modalities:
        name = modality.get("name")
        folder = modality.get("folder_name") or name
        if not name or not folder:
            continue
        dirs.append(ctx.workspace_root / "results" / name / folder)
    return dirs


def step_preproc(ctx: RunContext) -> int:
    preproc_cmd = ctx.config.get("pipeline", {}).get("preproc_command")
    if not preproc_cmd:
        print("[preproc] skipped (no pipeline.preproc_command in config)")
        return 0

    cmd = ["bash", "-lc", preproc_cmd]
    return _run_command(cmd, ctx, "preproc")


def step_stats(ctx: RunContext) -> int:
    cat12_dir = _resolve_cat12_dir(ctx)
    participants_file = _resolve_participants(ctx)

    if not cat12_dir:
        print("[stats] missing CAT12 dir. Provide --cat12-dir or set analysis.cat12_dir in config.")
        return 2

    script = ctx.workspace_root / "scripts" / "analysis" / "cat12_multi_modality.sh"
    cmd = [
        "bash",
        str(script),
        "--config",
        str(ctx.config_path),
        "--cat12-dir",
        str(cat12_dir),
    ]

    if participants_file:
        cmd.extend(["--participants", str(participants_file)])

    if ctx.modality:
        cmd.extend(["--modality", ctx.modality])

    if ctx.force:
        cmd.append("--force-all")

    return _run_command(cmd, ctx, "stats")


def step_sweep(ctx: RunContext) -> int:
    results_dirs = _resolve_results_dirs(ctx)
    if not results_dirs:
        print("[sweep] no results dirs resolved from config.analysis.modalities")
        return 2

    script = ctx.workspace_root / "scripts" / "analysis" / "run_stats_sweep.py"
    any_fail = False

    for results_dir in results_dirs:
        spm_mat = results_dir / "SPM.mat"
        if not spm_mat.exists() and not ctx.dry_run:
            print(f"[sweep] skip {results_dir} (missing SPM.mat)")
            continue

        cmd = [
            sys.executable,
            str(script),
            str(results_dir),
            "--config",
            str(ctx.config_path),
        ]
        if ctx.use_matlab:
            cmd.append("--use-matlab")
        if ctx.force:
            cmd.append("--force")

        rc = _run_command(cmd, ctx, "sweep")
        if rc != 0:
            any_fail = True

    return 1 if any_fail else 0


def step_report(ctx: RunContext) -> int:
    script = ctx.workspace_root / "scripts" / "reporting" / "auto_report.py"
    cmd = [sys.executable, str(script), str(ctx.config_path)]
    return _run_command(cmd, ctx, "report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified CAT12 pipeline runner (draft)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/pipeline/run_pipeline.py --config config/config_14_2_26.json\n"
            "  python scripts/pipeline/run_pipeline.py --config config/config_14_2_26.json --only stats,report\n"
            "  python scripts/pipeline/run_pipeline.py --config config/config_14_2_26.json --from-step sweep --until-step report\n"
            "  python scripts/pipeline/run_pipeline.py --config config/config_14_2_26.json --cat12-dir /path/to/cat12 --force\n"
            "  python scripts/pipeline/run_pipeline.py --config config/config_14_2_26.json --dry-run\n"
        ),
    )

    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--cat12-dir", help="CAT12 preprocessed folder")
    parser.add_argument("--participants", help="participants TSV path")
    parser.add_argument("--results-dir", help="Override results dir for sweep/report (single target)")
    parser.add_argument("--modality", help="Run stats only for one modality")

    parser.add_argument("--only", help="Comma-separated steps to run: preproc,stats,sweep,report")
    parser.add_argument("--skip", help="Comma-separated steps to skip")
    parser.add_argument("--from-step", help="Start at step")
    parser.add_argument("--until-step", help="Stop after step")

    parser.add_argument("--use-matlab", action="store_true", help="Pass --use-matlab to sweep step")
    parser.add_argument("--force", action="store_true", help="Force rerun where supported")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands only")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    utils_dir = workspace_root / "scripts" / "utils"
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = (workspace_root / config_path).resolve()

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 2

    try:
        from validate_config_json import validate_config_file  # type: ignore

        valid, validation_errors = validate_config_file(config_path)
        if not valid:
            print(f"Config schema validation failed: {config_path}")
            for err in validation_errors[:20]:
                print(f"  - {err}")
            if len(validation_errors) > 20:
                print(f"  ... and {len(validation_errors) - 20} more")
            return 2
    except Exception as exc:
        print(f"Config schema validation could not run: {exc}")
        return 2

    try:
        config = _load_json(config_path)
    except Exception as exc:
        print(f"Failed to parse config: {exc}")
        return 2

    try:
        steps = _select_steps(
            config=config,
            only=_parse_step_list(args.only),
            skip=_parse_step_list(args.skip),
            from_step=args.from_step,
            until_step=args.until_step,
        )
    except ValueError as exc:
        print(str(exc))
        return 2

    if not steps:
        print("No steps selected. Nothing to do.")
        return 0

    ctx = RunContext(
        workspace_root=workspace_root,
        config_path=config_path,
        config=config,
        cat12_dir=Path(args.cat12_dir).resolve() if args.cat12_dir else None,
        participants_file=Path(args.participants).resolve() if args.participants else None,
        modality=args.modality,
        results_dir_override=Path(args.results_dir).resolve() if args.results_dir else None,
        use_matlab=args.use_matlab,
        force=args.force,
        dry_run=args.dry_run,
    )

    handlers = {
        "preproc": step_preproc,
        "stats": step_stats,
        "sweep": step_sweep,
        "report": step_report,
    }

    print("Unified pipeline (draft)")
    print(f"Config: {ctx.config_path}")
    print(f"Steps: {', '.join(steps)}")
    if ctx.dry_run:
        print("Mode: dry-run")

    for step in steps:
        rc = handlers[step](ctx)
        if rc != 0:
            print(f"\nPipeline stopped at step '{step}' (exit code {rc})")
            return rc

    print("\nPipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
