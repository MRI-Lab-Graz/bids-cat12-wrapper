#!/usr/bin/env python3
"""Generate a minimal participants.tsv for stats from a BIDS dataset."""

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
from bids import BIDSLayout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a minimal participants.tsv (scan-level) from BIDS"
    )
    parser.add_argument("--bids-dir", required=True, help="Path to BIDS dataset")
    parser.add_argument("--out", required=True, help="Output participants.tsv path")
    parser.add_argument(
        "--group-col", default="group", help="Group column name (default: group)"
    )
    parser.add_argument(
        "--session-col",
        default="session",
        help="Session column name (default: session)",
    )
    parser.add_argument(
        "--group-mode",
        default="even-odd",
        choices=["even-odd", "single"],
        help="Group assignment strategy",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject labels (e.g., 01 02 03). If omitted, use all.",
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help="Optional session labels to include (e.g., test retest).",
    )
    return parser.parse_args()


def assign_group(subject: str, mode: str) -> str:
    if mode == "single":
        return "all"
    # even-odd based on numeric suffix if present
    digits = "".join(ch for ch in subject if ch.isdigit())
    if digits:
        return "A" if int(digits) % 2 == 1 else "B"
    return "A"


def normalize_subjects(subjects: Optional[List[str]]) -> Optional[List[str]]:
    if not subjects:
        return None
    out = []
    for s in subjects:
        s = s.replace("sub-", "")
        out.append(s)
    return out


def main() -> None:
    args = parse_args()
    bids_dir = Path(args.bids_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    layout = BIDSLayout(bids_dir, validate=False)
    subjects = layout.get_subjects()
    requested = normalize_subjects(args.subjects)
    if requested:
        subjects = [s for s in subjects if s in requested]

    rows = []
    for subject in subjects:
        sessions = layout.get_sessions(subject=subject)
        if args.sessions:
            sessions = [s for s in sessions if s in args.sessions]
        if not sessions:
            sessions = ["1"]
        for session in sessions:
            rows.append(
                {
                    "participant_id": f"sub-{subject}",
                    args.group_col: assign_group(subject, args.group_mode),
                    args.session_col: session,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
