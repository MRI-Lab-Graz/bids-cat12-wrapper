#!/usr/bin/env python3
"""Generate per-contrast TFCE full-run plans based on probe summaries.

Usage:
  tfce_full_run_plan.py summary.json metadata.json cc_threshold default_full_perm [min_full_perm]

The script reads the probe summary JSON (produced by tfce_summary_from_log.py)
plus the optional metadata JSON (from export_tfce_probe_metadata.m) and emits
one tab-separated line per contrast:

    <contrast>\t<method>\t<n_perm>\t<reason_csv>

This lightweight format can be consumed directly by tfce_two_stage.sh.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

MIN_FULL_PERM_DEFAULT = 100
LOW_DF_THRESHOLD = 120
VERY_LOW_DF_THRESHOLD = 60
LOW_DF_CAP = 2000
VERY_LOW_DF_CAP = 1000
SPARSE_CONTRAST_CAP = 2000
FEW_CONDITIONS_CAP = 2000


def _load_json(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_metadata_map(metadata_list) -> Dict[int, dict]:
    if not metadata_list:
        return {}
    mapping: Dict[int, dict] = {}
    for entry in metadata_list:
        con = _to_int(entry.get("contrast"))
        if con is None:
            continue
        mapping[con] = entry
    return mapping


def derive_plan_for_entry(
    entry: dict,
    meta: dict,
    cc_threshold: float,
    default_full_perm: int,
    min_full_perm: int,
) -> Tuple[int, str, int, List[str]]:
    con = _to_int(entry.get("contrast"))
    if con is None:
        return None  # type: ignore

    reasons: List[str] = []
    cc_val = _to_float(entry.get("probe_cc"))
    method = entry.get("chosen_full_method")
    if not method:
        method = "freedman-lane" if (cc_val is not None and cc_val < cc_threshold) else "smith"
        if cc_val is not None and cc_val < cc_threshold:
            reasons.append(f"cc<{cc_threshold}")

    perm_caps: List[int] = []

    if meta:
        erdf = _to_float(meta.get("error_df"))
        if erdf is not None:
            if erdf < VERY_LOW_DF_THRESHOLD:
                perm_caps.append(VERY_LOW_DF_CAP)
                if method != "freedman-lane":
                    method = "freedman-lane"
                reasons.append("erdf<60")
            elif erdf < LOW_DF_THRESHOLD:
                perm_caps.append(LOW_DF_CAP)
                reasons.append("erdf<120")
        nnz = _to_int(meta.get("nnz_weights"))
        if nnz is not None and nnz <= 4:
            perm_caps.append(SPARSE_CONTRAST_CAP)
            reasons.append("sparse_contrast")

    max_unique = _to_int(entry.get("max_unique_perms"))
    if max_unique is not None and max_unique > 0:
        perm_caps.append(max_unique)
        reasons.append(f"perm_cap_hint={max_unique}")

    conditions = _to_int(entry.get("conditions_count"))
    if conditions is not None and conditions <= 4:
        perm_caps.append(FEW_CONDITIONS_CAP)
        reasons.append("few_conditions")

    if entry.get("equal_sample_hint"):
        reasons.append("equal_sample_half")

    perm_target = default_full_perm
    if perm_caps:
        perm_target = min([default_full_perm] + perm_caps)

    min_cap = min(perm_caps) if perm_caps else None
    if (min_cap is None or min_cap >= min_full_perm) and perm_target < min_full_perm:
        perm_target = min_full_perm

    perm_target = max(1, int(round(perm_target)))

    return con, method, perm_target, reasons


def main() -> int:
    if len(sys.argv) < 5:
        print(__doc__)
        return 2

    summary_path = sys.argv[1]
    metadata_path = sys.argv[2]
    try:
        cc_threshold = float(sys.argv[3])
    except ValueError:
        print("Invalid cc_threshold", file=sys.stderr)
        return 2
    default_full_perm = _to_int(sys.argv[4])
    if default_full_perm is None or default_full_perm <= 0:
        print("Invalid default full permutation count", file=sys.stderr)
        return 2
    if len(sys.argv) > 5:
        min_full_perm = max(1, _to_int(sys.argv[5]) or MIN_FULL_PERM_DEFAULT)
    else:
        min_full_perm = MIN_FULL_PERM_DEFAULT

    summary = _load_json(summary_path)
    if not summary:
        return 0
    metadata_obj = _load_json(metadata_path)
    metadata_map = {}
    if isinstance(metadata_obj, list):
        metadata_map = build_metadata_map(metadata_obj)

    plans: List[Tuple[int, str, int, List[str]]] = []
    for entry in summary:
        plan = derive_plan_for_entry(
            entry,
            metadata_map.get(_to_int(entry.get("contrast")), {}),
            cc_threshold,
            default_full_perm,
            min_full_perm,
        )
        if plan is None:
            continue
        plans.append(plan)

    plans.sort(key=lambda item: item[0])

    for con, method, perms, reasons in plans:
        reason_str = ",".join(reasons) if reasons else "-"
        print(f"{con}\t{method}\t{perms}\t{reason_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
