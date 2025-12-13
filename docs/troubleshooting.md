# Troubleshooting

## Installation

### Installer created files in the wrong place

Expected after install:
- `.env` and `.venv/` at repo root
- `external/` at repo root

If you see `scripts/external/` or `scripts/.env`, you likely ran an older installer. Re-run:

```bash
./scripts/install_cat12_standalone.sh
```

and ensure you run it from the repo root.

### `deno: command not found`

The BIDS validator wrapper may require Deno.

- Re-run the installer to install a workspace-local Deno.
- Or skip validation with preprocessing: `--no-validate`.

## Preprocessing

### "No processing stages specified"

You must specify at least one stage:

```bash
./cat12_prepro /data/bids /data/derivatives/cat12 participant --preproc
```

### `CAT12_ROOT` / `MCR_ROOT` not set

- Run the installer.
- Or `source activate_cat12.sh`.

### Runs out of RAM

- Reduce `--n-jobs`.
- Avoid running surface processing if you only need VBM (`--no-surface`).

## Statistics

### MATLAB / SPM errors

`cat12_stats` can run either:
- via system MATLAB + SPM (if configured), or
- via standalone mode (if MATLAB isn’t found)

Check defaults in `config/config.ini` and confirm:
- `MATLAB.exe` exists (if you intend to use MATLAB)
- `SPM.path` is correct (optional)

### Results directory already exists

Use `--force` to overwrite, or set `force_clean = true` in `config/config.ini`.

### TFCE takes too long

- Use `--pilot` first.
- Reduce permutations with `--n-perm`.
- Increase parallelism with `--n-jobs` if CPU/RAM allow.
