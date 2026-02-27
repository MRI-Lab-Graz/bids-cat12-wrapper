#!/usr/bin/env python3
"""Validate CAT12 stats config JSON against JSON Schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple


def _default_schema_path() -> Path:
    workspace_root = Path(__file__).resolve().parents[2]
    return workspace_root / "config" / "schema" / "config.schema.json"


def validate_config_file(config_path: Path, schema_path: Path | None = None) -> Tuple[bool, List[str]]:
    config_path = Path(config_path)
    schema_path = Path(schema_path) if schema_path else _default_schema_path()

    errors: List[str] = []

    if not config_path.exists():
        return False, [f"Config file not found: {config_path}"]

    if not schema_path.exists():
        return False, [f"Schema file not found: {schema_path}"]

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config_data = json.load(handle)
    except Exception as exc:
        return False, [f"Failed to parse config JSON: {exc}"]

    try:
        with schema_path.open("r", encoding="utf-8") as handle:
            schema_data = json.load(handle)
    except Exception as exc:
        return False, [f"Failed to parse schema JSON: {exc}"]

    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return False, [
            "Missing dependency: jsonschema",
            "Install with: pip install jsonschema",
            "or install project requirements: pip install -r requirements.txt",
        ]

    validator = Draft202012Validator(schema_data)
    validation_errors = sorted(validator.iter_errors(config_data), key=lambda e: list(e.path))

    for err in validation_errors:
        path = ".".join(str(part) for part in err.path)
        if path:
            errors.append(f"{path}: {err.message}")
        else:
            errors.append(err.message)

    return len(errors) == 0, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CAT12 config JSON")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument(
        "--schema",
        default=None,
        help="Path to schema JSON (default: config/schema/config.schema.json)",
    )
    args = parser.parse_args()

    ok, errors = validate_config_file(Path(args.config), Path(args.schema) if args.schema else None)

    if ok:
        print(f"✓ Config is valid: {args.config}")
        return 0

    print(f"✗ Config validation failed: {args.config}")
    for err in errors:
        print(f"  - {err}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
