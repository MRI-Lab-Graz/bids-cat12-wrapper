# CAT12 Web UI

Local web interface for config editing + pipeline execution.

## Start

```bash
/Volumes/Thunder/129_PK01/cat12/stats/.venv/bin/python run_webui.py
```

The browser opens automatically at `http://127.0.0.1:5055`.

## What it provides

- Dynamic editor rendering all config JSON fields
- Pipeline run options (`only`, `skip`, `from-step`, `until-step`, paths, flags)
- Live web-terminal console with stdout/stderr stream
- Save/load reusable `project.json`

## Project persistence

Default project file:

- `projects/webui/project.json`

When you run the pipeline from UI, the app also writes:

- `projects/webui/runtime_config.json`

This runtime config is validated against schema before execution.

## Architecture

- Backend: `scripts/webui/app.py`
- Templates: `scripts/webui/templates/` (base + page + components)
- Styles: `scripts/webui/static/css/` (base/layout/forms/console)
- JavaScript: `scripts/webui/static/js/` (api/config-editor/runner/terminal/app)
