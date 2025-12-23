#!/usr/bin/env python3
"""
ml_batch_runner.py

Run multiple ML jobs (calls to utils/vbm_ml_interaction.py) driven by a JSON config file.
Writes per-job logs and an aggregated summary (CSV + JSON).

Usage:
  python utils/ml_batch_runner.py --config utils/ml_batch_config_example.json --outdir results/vbm/ml_batch_runs

Config schema (simple):
{
  "global": {
    "python_executable": ".venv_ml/bin/python",
    "script": "utils/vbm_ml_interaction.py",
    "mask": "templates/brainmask_GMtight.nii",
    "participants_tsv": "participants.tsv",
    "data_root": "/path/to/data"
  },
  "jobs": [ { ... }, ... ]
}

Each job may include fields that map to CLI flags of vbm_ml_interaction.py, for example:
  name, delta_type, session_a, session_b, classifier, k_best, n_permutations, n_jobs, use_unsmoothed, merge_interventions

The runner will:
 - create an output subdir for each job under the provided --outdir (job.name)
 - skip a job if its ml_summary.json already exists (resume behaviour)
 - write job stdout/stderr to job.log
 - collect summaries from ml_summary.json into an aggregated CSV/JSON at the end

"""

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from itertools import product
from pathlib import Path


def build_cmd(global_cfg, job_cfg, outdir_job):
    """Build command list from config dicts."""
    py = global_cfg.get('python_executable', 'python')
    script = global_cfg.get('script', 'utils/vbm_ml_interaction.py')

    # If paths are relative and not found as-is, attempt to resolve them
    # by searching upward from the current working directory and the
    # runner script directory. This helps when users specify paths like
    # ".venv_ml/bin/python" relative to the project root.
    def _resolve_if_needed(path_str):
        p = Path(path_str)
        if p.is_absolute() and p.exists():
            return str(p)
        if p.exists():
            # relative path that happens to exist from current cwd -> return absolute
            try:
                return str(p.resolve())
            except Exception:
                return str(p)
        # search upward from cwd
        for ancestor in [Path.cwd()] + list(Path.cwd().parents):
            cand = ancestor / path_str
            if cand.exists():
                return str(cand)
        # search upward from runner script directory
        runner_dir = Path(__file__).resolve().parent
        for ancestor in [runner_dir] + list(runner_dir.parents):
            cand = ancestor / path_str
            if cand.exists():
                return str(cand)
        return path_str

    py = _resolve_if_needed(py)
    script = _resolve_if_needed(script)
    cmd = [py, script]

    # common global args (if present)
    if global_cfg.get('mask'):
        cmd += ['--mask', global_cfg['mask']]
    if global_cfg.get('participants_tsv'):
        cmd += ['--participants-tsv', global_cfg['participants_tsv']]
    if global_cfg.get('data_root'):
        cmd += ['--data-root', global_cfg['data_root']]

    # output
    cmd += ['--output', str(outdir_job)]

    # now job-specific flags mapping
    def add_flag(flag, val):
        if val is None:
            return
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(val)])

    # simple mappings
    mappings = {
        'delta_type': '--delta-type',
        'session_a': '--session-a',
        'session_b': '--session-b',
        'classifier': '--classifier',
        'k_best': '--k-best',
        'n_permutations': '--n-permutations',
        'n_jobs': '--n-jobs',
        'group_col': '--group-col'
    }
    # allow controlling voxel filtering via config: min_finite_prop maps to
    # the --min-finite-prop CLI flag of vbm_ml_interaction.py
    mappings['min_finite_prop'] = '--min-finite-prop'
    # allow controlling classifier class_weight and CV folds from config
    mappings['class_weight'] = '--class-weight'
    mappings['cv_folds'] = '--cv-folds'
    for k, flag in mappings.items():
        # prefer job-level override, otherwise fall back to global config
        val = job_cfg.get(k, None)
        if val is None:
            val = global_cfg.get(k, None)
        if val is not None:
            add_flag(flag, val)

    # boolean flags: only respect job-level boolean flags here. We intentionally
    # do not forward a global `merge_interventions` setting to avoid implicit
    # behaviour; users should set boolean flags on a per-job basis in the config.
    for bool_key, flagname in [('use_unsmoothed', '--use-unsmoothed'), ('merge_interventions', '--merge-interventions')]:
        if job_cfg.get(bool_key):
            cmd.append(flagname)

    # optional participants override
    if job_cfg.get('participants_tsv') and job_cfg.get('data_root'):
        # prefer job-specific participants/data root if provided
        cmd += ['--participants-tsv', job_cfg['participants_tsv'], '--data-root', job_cfg['data_root']]

    return cmd


def run_jobs(config_path, outdir, resume=True, dry_run=False, two_phase=False, phase1_perms=50, phase2_perms=5000, promote_pvalue_threshold=0.05, promote_pvalue_margin=0.02, promote_top_k=0):
    cfg = json.load(open(config_path))
    global_cfg = cfg.get('global', {})
    jobs = cfg.get('jobs', [])

    # Resolve relative paths for python_executable and script so commands
    # work even when the runner is invoked from a different working dir.
    # Try (in order): as given, relative to config file directory, relative
    # to this runner script's directory.
    cfg_dir = Path(config_path).resolve().parent
    runner_dir = Path(__file__).resolve().parent
    for key in ('python_executable', 'script'):
        val = global_cfg.get(key)
        if not val:
            continue
        p = Path(val)
        if p.is_absolute() and p.exists():
            # absolute and exists -> keep
            continue
        found = False
        # Search upward from the config directory through ancestors so
        # project-root-relative entries like ".venv_ml/bin/python" are found.
        for ancestor in [cfg_dir] + list(cfg_dir.parents):
            cand = ancestor / val
            if cand.exists():
                global_cfg[key] = str(cand)
                found = True
                break
        if found:
            continue
        # fallback: try runner script directory
        cand2 = runner_dir / val
        if cand2.exists():
            global_cfg[key] = str(cand2)
            continue

    # support 'sweeps' shorthand: each sweep is a dict with 'name_prefix' and
    # a mapping of parameter keys to lists of possible values. We'll expand
    # the cartesian product into concrete job dicts and append to jobs.
    sweeps = cfg.get('sweeps', [])
    for sweep in sweeps:
        name_prefix = sweep.get('name_prefix', 'sweep')
        params = sweep.get('params', {})
        # sort keys to make expansion deterministic
        keys = sorted(params.keys())
        lists = [params[k] for k in keys]
        for idx, combo in enumerate(product(*lists)):
            job = {}
            for k, v in zip(keys, combo):
                job[k] = v
            # create a sensible name
            name = sweep.get('name_format')
            if name:
                try:
                    job_name = name.format(idx=idx, **job)
                except Exception:
                    job_name = f"{name_prefix}_{idx}"
            else:
                # default name: prefix + key=value pairs
                parts = [f"{k}={str(job[k]).replace(' ', '')}" for k in keys]
                job_name = name_prefix + '_' + '_'.join(parts)
            job['name'] = job_name
            jobs.append(job)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summaries = []

    def execute_jobs(job_list, phase_label='phase'):
        phase_summaries = []
        for job in job_list:
            name = job.get('name') or f"job_{int(time.time())}"
            job_dir = outdir / name
            job_dir.mkdir(parents=True, exist_ok=True)
            summary_path = job_dir / 'ml_summary.json'
            log_path = job_dir / 'job.log'

            if resume and summary_path.exists():
                print(f"Skipping {name}: summary exists ({summary_path})")
                try:
                    summary = json.load(open(summary_path))
                    summary['_job'] = name
                    summary['_outdir'] = str(job_dir)
                    phase_summaries.append(summary)
                except Exception:
                    print('  Warning: failed to read existing summary; will re-run')
                continue

            cmd = build_cmd(global_cfg, job, job_dir)
            print(f'Running {phase_label} job:', name)
            print('  cmd:', ' '.join(shlex.quote(c) for c in cmd))
            if dry_run:
                continue

            # Run subprocess and capture FileNotFoundError (common when executable
            # path is incorrect). Write stdout/stderr to log file.
            try:
                with open(log_path, 'wb') as logf:
                    proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
                if proc.returncode != 0:
                    print(f'Job {name} failed (returncode {proc.returncode}); see {log_path}')
                    # still attempt to capture partial summary if any
                    if summary_path.exists():
                        try:
                            summary = json.load(open(summary_path))
                            summary['_job'] = name
                            summary['_outdir'] = str(job_dir)
                            phase_summaries.append(summary)
                        except Exception:
                            pass
                    continue
            except FileNotFoundError as e:
                print(f'Job {name} failed to start: executable not found: {e}')
                print('  Tried to run:', cmd[0])
                print('  Suggestion: ensure `python_executable` in the config is an absolute path or exists relative to the project root.')
                # write the error to the log file for debugging
                with open(log_path, 'w') as logf:
                    logf.write(f'Failed to start job: {e}\n')
                    logf.write('Command: ' + ' '.join(shlex.quote(c) for c in cmd) + '\n')
                continue
            except OSError as e:
                print(f'Job {name} failed to start due to OS error: {e}')
                with open(log_path, 'w') as logf:
                    logf.write(f'Failed to start job (OSError): {e}\n')
                    logf.write('Command: ' + ' '.join(shlex.quote(c) for c in cmd) + '\n')
                continue

            # try to load ml_summary.json
            if summary_path.exists():
                try:
                    summary = json.load(open(summary_path))
                    summary['_job'] = name
                    summary['_outdir'] = str(job_dir)
                    phase_summaries.append(summary)
                except Exception as e:
                    print(f'  Warning: could not parse summary for {name}: {e}')
            else:
                print(f'  Warning: job finished but no summary at {summary_path}')
        return phase_summaries

    # ensure every job has a name
    for job in jobs:
        if 'name' not in job:
            job['name'] = f"job_{int(time.time())}"

    # Phase 1: run all jobs with low permutations (unless n_permutations explicitly set and two_phase is False)
    phase1_jobs = []
    for job in jobs:
        j = dict(job)
        if two_phase:
            j['n_permutations'] = phase1_perms
        else:
            # respect job-specified n_permutations if present
            if 'n_permutations' not in j:
                j['n_permutations'] = phase1_perms
        phase1_jobs.append(j)

    summaries_phase1 = execute_jobs(phase1_jobs, phase_label='phase1')
    summaries.extend(summaries_phase1)

    # If two_phase requested, select candidates to promote and run phase2
    if two_phase:
        promote_candidates = []
        # map original job name to original job spec
        job_map = {job['name']: job for job in jobs}
        for s in summaries_phase1:
            jobname = s.get('_job')
            if not jobname:
                continue
            pval = s.get('permutation_pvalue')
            if pval is None:
                continue
            if pval <= (promote_pvalue_threshold + promote_pvalue_margin):
                # promote: build a phase2 job with higher permutations
                orig = job_map.get(jobname, None)
                if orig is None:
                    continue
                newjob = dict(orig)
                newjob['name'] = f"{jobname}_phase2"
                newjob['n_permutations'] = phase2_perms
                promote_candidates.append(newjob)

        # optionally also promote top-k by accuracy if requested
        if promote_top_k and promote_top_k > 0:
            # sort by cv_accuracy_mean descending
            scored = [s for s in summaries_phase1 if 'cv_accuracy_mean' in s]
            scored_sorted = sorted(scored, key=lambda x: float(x.get('cv_accuracy_mean', 0.0)), reverse=True)
            for s in scored_sorted[:promote_top_k]:
                jobname = s.get('_job')
                if not jobname:
                    continue
                orig = job_map.get(jobname, None)
                if orig is None:
                    continue
                nominee_name = f"{jobname}_phase2"
                if any(n['name'] == nominee_name for n in promote_candidates):
                    continue
                newjob = dict(orig)
                newjob['name'] = nominee_name
                newjob['n_permutations'] = phase2_perms
                promote_candidates.append(newjob)

        if promote_candidates:
            print(f'Promoting {len(promote_candidates)} candidates to phase2 (n_permutations={phase2_perms})')
            summaries_phase2 = execute_jobs(promote_candidates, phase_label='phase2')
            summaries.extend(summaries_phase2)

    # write aggregated outputs
    agg_json = outdir / 'aggregated_summaries.json'
    json.dump(summaries, open(agg_json, 'w'), indent=2)

    # also write CSV
    agg_csv = outdir / 'aggregated_summaries.csv'
    if summaries:
        # collect union of keys
        keys = set()
        for s in summaries:
            keys.update(s.keys())
        keys = sorted(keys)
        with open(agg_csv, 'w', newline='') as cf:
            writer = csv.DictWriter(cf, fieldnames=keys)
            writer.writeheader()
            for s in summaries:
                writer.writerow({k: s.get(k, '') for k in keys})
    print('Wrote aggregated results to', agg_json, 'and', agg_csv)
    return summaries


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Batch runner for vbm_ml_interaction jobs')
    # If invoked with no arguments, show help and exit (more friendly than argparse's
    # default error about missing required arguments).
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    p.add_argument('--config', required=True, help='Path to JSON config (see utils/ml_batch_config_example.json)')
    p.add_argument('--outdir', required=True, help='Directory to store per-job outputs')
    p.add_argument('--no-resume', dest='resume', action='store_false', help='Do not skip jobs that have existing ml_summary.json')
    p.add_argument('--dry-run', action='store_true', help='Show commands but do not execute')
    p.add_argument('--two-phase', action='store_true', help='Run a two-phase workflow: low-permutation sweep then promote promising jobs to high-permutation re-runs')
    p.add_argument('--phase1-perms', type=int, default=50, help='Number of permutations for phase1 (coarse scan)')
    p.add_argument('--phase2-perms', type=int, default=5000, help='Number of permutations for phase2 (confirmation)')
    p.add_argument('--promote-pvalue-threshold', type=float, default=0.05, help='P-value threshold for promotion')
    p.add_argument('--promote-pvalue-margin', type=float, default=0.02, help='Additional margin above threshold to promote (useful for borderline candidates)')
    p.add_argument('--promote-top-k', type=int, default=0, help='Additionally promote top-K jobs by accuracy regardless of p-value (0 = disabled)')
    args = p.parse_args()

    run_jobs(
        args.config,
        args.outdir,
        resume=args.resume,
        dry_run=args.dry_run,
        two_phase=args.two_phase,
        phase1_perms=args.phase1_perms,
        phase2_perms=args.phase2_perms,
        promote_pvalue_threshold=args.promote_pvalue_threshold,
        promote_pvalue_margin=args.promote_pvalue_margin,
        promote_top_k=args.promote_top_k,
    )
