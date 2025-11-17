#!/usr/bin/env python3
"""Parse a cleaned MATLAB TFCE log and produce a per-contrast summary JSON.

Usage:
  tfce_summary_from_log.py <logfile> <out_json> [--cc-threshold FLOAT]

The script extracts for each contrast the last seen 'cc' value (if any),
screening warnings, equal-sample hints, and the probe/full nuisance method
phrases printed by the wrapper (e.g., 'Probe nuisance method: ...' and
'Using nuisance method: ...'). It writes a JSON list of objects with fields:
    contrast: int
    probe_cc: float|null
    probe_warning: string|null
    probe_method: string|null
    chosen_full_method: string ("smith" or "freedman-lane")
    equal_sample_hint: bool
    max_unique_perms: int|null (from TFCE stopping diagnostics)
    permutation_stop_reason: string|null
    conditions_count: int|null (# of conditions reported by TFCE)
    exchangeability: string|null (verbatim TFCE line)

"""
import json
import re
import sys
from collections import OrderedDict
from typing import List, Tuple

CC_PATTERN = r"cc=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def _split_blocks(txt: str) -> List[Tuple[int, str]]:
    blocks: List[Tuple[int, str]] = []

    # Preferred format (new wrapper): '=== Contrast N ==='
    marker = re.compile(r"^=== Contrast\s+([0-9]+)\s+===\s*$", re.M)
    matches = list(marker.finditer(txt))
    if matches:
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(txt)
            blocks.append((int(match.group(1)), txt[start:end]))
        return blocks

    # Fallback format (MATLAB log only): 'Use contrast #N ...'
    use_contrast = re.compile(r"Use contrast #([0-9]+)[^\n]*", re.M)
    matches = list(use_contrast.finditer(txt))
    if not matches:
        return []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(txt)
        blocks.append((int(match.group(1)), txt[start:end]))
    return blocks


def _extract_cc(block: str):
    cc_matches = re.findall(CC_PATTERN, block, flags=re.I)
    if not cc_matches:
        return None
    try:
        return float(cc_matches[-1])
    except ValueError:
        return None


def _extract_warning(block: str):
    warn_match = re.search(
        r"(WARNING:.*?" + CC_PATTERN + r".*?)",
        block,
        flags=re.I | re.S,
    )
    if not warn_match:
        warn_match = re.search(
            r"(Correlation between[^\n]*" + CC_PATTERN + r"[^\n]*)",
            block,
            flags=re.I,
        )
    if not warn_match:
        return None
    # Collapse whitespace to a single line for readability
    return re.sub(r"\s+", " ", warn_match.group(1)).strip()


def _extract_equal_sample(block: str) -> bool:
    return bool(
        re.search(
            r"Equal sample sizes:\s*Use half the number of permutations",
            block,
        )
    )


def _extract_conditions(block: str):
    cond_match = re.search(r"# of conditions:\s*([0-9]+)", block)
    if not cond_match:
        return None
    try:
        return int(cond_match.group(1))
    except ValueError:
        return None


def _extract_exchangeability(block: str):
    exch_match = re.search(
        r"Exchangeability block/variable:\s*([^\n]+)", block
    )
    if not exch_match:
        return None
    return exch_match.group(1).strip()


def _extract_perm_stop(block: str):
    stop_match = re.search(
        r"Stopped after\s*([0-9]+)\s*permutations([^\n]*)",
        block,
    )
    if not stop_match:
        return None, None
    try:
        max_perm = int(stop_match.group(1))
    except ValueError:
        max_perm = None
    reason = stop_match.group(0).strip()
    return max_perm, reason


def parse_log(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    results = []
    blocks = _split_blocks(txt)
    if not blocks:
        return results

    latest_blocks: OrderedDict[int, str] = OrderedDict()
    for idx, block in blocks:
        if idx in latest_blocks:
            del latest_blocks[idx]
        latest_blocks[idx] = block

    for idx, block in latest_blocks.items():
        probe_cc = _extract_cc(block)
        probe_warning = _extract_warning(block)
        equal_sample_hint = _extract_equal_sample(block)
        conditions = _extract_conditions(block)
        exchangeability = _extract_exchangeability(block)
        max_perm, stop_reason = _extract_perm_stop(block)

        probe_method_match = re.search(r"Probe nuisance method:\s*(\S+)", block)
        probe_method = probe_method_match.group(1) if probe_method_match else None

        chosen_full_match = re.search(r"Using nuisance method:\s*(\S+)", block)
        chosen_full = chosen_full_match.group(1) if chosen_full_match else None

        results.append(
            {
                "contrast": idx,
                "probe_cc": probe_cc,
                "probe_warning": probe_warning,
                "probe_method": probe_method,
                "chosen_full_method": chosen_full,
                "equal_sample_hint": equal_sample_hint,
                "conditions_count": conditions,
                "exchangeability": exchangeability,
                "max_unique_perms": max_perm,
                "permutation_stop_reason": stop_reason,
            }
        )

    return results

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    logfile = sys.argv[1]
    outjson = sys.argv[2]
    data = parse_log(logfile)
    with open(outjson, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Wrote summary for {len(data)} contrasts to {outjson}")

if __name__ == '__main__':
    main()
