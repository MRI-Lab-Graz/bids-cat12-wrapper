# Unified Pipeline Draft (Single Entry Script)

This draft introduces one orchestrator command:

```bash
python scripts/pipeline/run_pipeline.py --config <config.json>
```

## Design Goals

- One script + one JSON config
- No flags => run all possible enabled steps
- Flags => select, skip, or slice steps
- Wrap existing scripts (no workflow break)
- Validate config JSON against schema before execution

## Config Schema

- Schema path: `config/schema/config.schema.json`
- Validator utility: `scripts/utils/validate_config_json.py`
- The unified runner and stats preflight both fail fast if config validation fails.

Validate manually:

```bash
python3 scripts/utils/validate_config_json.py --config config/config_14_2_26.json
```

## Supported Steps (current draft)

Order is fixed:

1. `preproc` (optional, only if configured)
2. `stats`
3. `sweep`
4. `report`

## Flag Model

- `--only preproc,stats,sweep,report`
- `--skip <comma-list>`
- `--from-step <step>`
- `--until-step <step>`
- `--dry-run`
- `--force`

Plus runtime paths:

- `--cat12-dir`
- `--participants`
- `--results-dir`
- `--modality`

## Examples

Run everything possible from config:

```bash
python scripts/pipeline/run_pipeline.py --config config/config_14_2_26.json
```

Run only stats + reporting:

```bash
python scripts/pipeline/run_pipeline.py \
  --config config/config_14_2_26.json \
  --only stats,report \
  --cat12-dir /path/to/cat12
```

Run from sweep onward:

```bash
python scripts/pipeline/run_pipeline.py \
  --config config/config_14_2_26.json \
  --from-step sweep
```

Preview commands only:

```bash
python scripts/pipeline/run_pipeline.py \
  --config config/config_14_2_26.json \
  --dry-run
```

## Optional Config Hook for Preprocessing

The draft supports a simple preproc hook:

```json
{
  "pipeline": {
    "preproc_command": "./cat12_prepro ...",
    "steps": {
      "preproc": {"enabled": true},
      "stats": {"enabled": true},
      "sweep": {"enabled": true},
      "report": {"enabled": true}
    }
  }
}
```

If `pipeline.preproc_command` is missing, `preproc` is skipped automatically.

## Current Wrapping Behavior

- `stats` => `scripts/analysis/cat12_multi_modality.sh`
- `sweep` => `scripts/analysis/run_stats_sweep.py` (per configured modality)
- `report` => `scripts/reporting/auto_report.py`

## Next MVP Improvements

- Add `--resume` with state file (`runs/<timestamp>/state.json`)
- Add `--continue-on-error` mode for batch runs
- Validate config keys against a lightweight schema before execution
- Add short alias script at repo root (e.g. `./pipeline`)
