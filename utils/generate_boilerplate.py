"""
Generate a reproducible boilerplate summary for CAT12 BIDS processing runs.
Outputs both Markdown and HTML files with system, software, and run details.
"""
import os
import sys
import platform
import socket
import json
import yaml
import datetime
import subprocess
from pathlib import Path

def get_spm_cat_version(spm_path):
    # Try to get SPM12 and CAT12 version from the standalone script
    try:
        result = subprocess.run([
            "grep", "SPM12", spm_path
        ], capture_output=True, text=True)
        spm_line = result.stdout.strip()
        result = subprocess.run([
            "grep", "CAT12", spm_path
        ], capture_output=True, text=True)
        cat_line = result.stdout.strip()
        return spm_line, cat_line
    except Exception:
        return "SPM12 version: unknown", "CAT12 version: unknown"

def get_system_info():
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "hostname": socket.gethostname(),
        "cpu": platform.processor(),
        "ram_gb": round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.**3), 2)
    }

def get_env_vars():
    keys = ["LD_LIBRARY_PATH", "SPM12_PATH", "CAT12_PATH"]
    return {k: os.environ.get(k, "") for k in keys}

def load_config(config_path):
    if config_path.endswith(".json"):
        with open(config_path) as f:
            return json.load(f)
    elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}

def render_markdown(info):
    md = f"""# CAT12 BIDS Processing Boilerplate

**Date:** {info['date']}
**Host:** {info['system']['hostname']}
**OS:** {info['system']['os']}
**Python:** {info['system']['python']}
**CPU:** {info['system']['cpu']}
**RAM:** {info['system']['ram_gb']} GB

---

**SPM12 Version:** {info['spm_version']}
**CAT12 Version:** {info['cat_version']}

---

**CLI Arguments:**
```
{info['cli_args']}
```

**Config File:** `{info['config_path']}`
```json
{json.dumps(info['config'], indent=2)}
```

**Environment Variables:**
```
{info['env_vars']}
```

**Input Directory:** `{info['input_dir']}`
**Output Directory:** `{info['output_dir']}`
**Subjects:** {info['subjects']}
**Sessions:** {info['sessions']}

---

*This boilerplate was auto-generated for reproducibility.*
"""
    return md

def render_html(info):
    html = f"""
<html><head><title>CAT12 BIDS Processing Boilerplate</title></head><body>
<h1>CAT12 BIDS Processing Boilerplate</h1>
<ul>
<li><b>Date:</b> {info['date']}</li>
<li><b>Host:</b> {info['system']['hostname']}</li>
<li><b>OS:</b> {info['system']['os']}</li>
<li><b>Python:</b> {info['system']['python']}</li>
<li><b>CPU:</b> {info['system']['cpu']}</li>
<li><b>RAM:</b> {info['system']['ram_gb']} GB</li>
</ul>
<hr>
<ul>
<li><b>SPM12 Version:</b> {info['spm_version']}</li>
<li><b>CAT12 Version:</b> {info['cat_version']}</li>
</ul>
<hr>
<b>CLI Arguments:</b><pre>{info['cli_args']}</pre>
<b>Config File:</b> {info['config_path']}<pre>{json.dumps(info['config'], indent=2)}</pre>
<b>Environment Variables:</b><pre>{info['env_vars']}</pre>
<b>Input Directory:</b> {info['input_dir']}<br>
<b>Output Directory:</b> {info['output_dir']}<br>
<b>Subjects:</b> {info['subjects']}<br>
<b>Sessions:</b> {info['sessions']}<br>
<hr>
<i>This boilerplate was auto-generated for reproducibility.</i>
</body></html>
"""
    return html

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate CAT12 BIDS boilerplate summary.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subjects", nargs='+', required=True)
    parser.add_argument("--sessions", nargs='+', default=None)
    parser.add_argument("--cli-args", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--spm-script", required=True)
    args = parser.parse_args()

    info = {}
    info['date'] = datetime.datetime.now().isoformat()
    info['system'] = get_system_info()
    spm_version, cat_version = get_spm_cat_version(args.spm_script)
    info['spm_version'] = spm_version
    info['cat_version'] = cat_version
    info['cli_args'] = args.cli_args
    info['config_path'] = args.config_path
    info['config'] = load_config(args.config_path)
    info['env_vars'] = get_env_vars()
    info['input_dir'] = args.input_dir
    info['output_dir'] = args.output_dir
    info['subjects'] = ', '.join(args.subjects)
    info['sessions'] = ', '.join(args.sessions) if args.sessions else 'N/A'

    md = render_markdown(info)
    html = render_html(info)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, "boilerplate.md"), "w") as f:
        f.write(md)
    with open(os.path.join(args.output_dir, "boilerplate.html"), "w") as f:
        f.write(html)
    print(f"Boilerplate written to {args.output_dir}/boilerplate.md and .html")

if __name__ == "__main__":
    main()
