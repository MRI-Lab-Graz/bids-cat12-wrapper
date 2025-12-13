#!/usr/bin/env python
"""Compatibility entrypoint.

This repository's main preprocessing CLI lives in:
  scripts/preprocessing/bids_cat12_processor.py

Some docs/tools may expect a top-level module named `bids_cat12_processor`.
This file forwards to the Click command implementation.
"""

from scripts.preprocessing.bids_cat12_processor import main


if __name__ == "__main__":
    # Ensure a stable command name in help text.
    main(prog_name="cat12_prepro")
