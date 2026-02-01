#!/usr/bin/env python3
"""
Generate a compact ASCII preview of a design.json design matrix.
Writes `design_ascii.txt` next to the design.json (or to an explicit --output file).
Prints the first N lines to stdout for quick preview.
"""
import json
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='Print ASCII design matrix preview from design.json')
parser.add_argument('design', nargs='?', help='Path to design.json (default: ./design.json)', default='design.json')
parser.add_argument('--output', '-o', help='Output ASCII file path (default: <design_dir>/design_ascii.txt)')
parser.add_argument('--rows', '-n', type=int, default=20, help='Number of rows to include in terminal preview')
parser.add_argument('--width', '-w', type=int, default=80, help='Terminal width for ASCII matrix')
args = parser.parse_args()

p = Path(args.design)
if not p.exists():
    # try searching upward
    candidates = list(Path('.').rglob('design.json'))
    if candidates:
        # prefer results/vbm path
        candidates_sorted = sorted(candidates, key=lambda x: ('/results/vbm/' in str(x), x.stat().st_mtime))
        p = candidates_sorted[-1]

if not p.exists():
    raise SystemExit(f'design.json not found (tried {args.design})')

data = json.loads(p.read_text())
files = data.get('files', [])
covs = data.get('covariates', {})
groups = data.get('groups', {})
sessions = data.get('sessions', [])

# Determine unique cell columns for group_by_session in stable order
unique_cells = []
seen = set()
for f in files:
    uc = f"{f.get('group')}_by_{f.get('session')}"
    if uc not in seen:
        seen.add(uc)
        unique_cells.append(uc)

# Build ascii rows limited to args.width
ncols = len(unique_cells) + len(covs)
# We'll allocate columns: block for unique_cells, then a small block for covariates
cell_area = min(len(unique_cells), max(1, int((args.width-20)*0.6)))
cov_area = max(4, min(len(covs), args.width - cell_area - 20))
# fallback column char width
col_w = 1

# Helper to render a row
def render_row(idx, file_entry, cov_lists, cell_map):
    # start with row index, subject, session
    left = f"{idx:>3} {file_entry.get('subject')[:12]:12} s{file_entry.get('session')}"
    # cell block
    cell_block = []
    for i,uc in enumerate(unique_cells[:cell_area]):
        val = '█' if cell_map.get(uc, [])[idx] else ' '
        cell_block.append(val)
    # cov block
    cov_block = []
    for i,cname in enumerate(list(cov_lists.keys())[:cov_area]):
        try:
            _ = cov_lists[cname][idx]
            cov_block.append('#')
        except Exception:
            cov_block.append(' ')
    return left + ' ' + ''.join(cell_block) + '  ' + ''.join(cov_block)

# Build cell_map as lists for quick access
cell_map = {}
for uc in unique_cells:
    cell_map[uc] = []
for f in files:
    uc = f"{f.get('group')}_by_{f.get('session')}"
    for key in unique_cells:
        cell_map[key].append(1 if key==uc else 0)

cov_lists = {k:list(v) for k,v in covs.items()}

# Compose output lines
lines = []
header = '   subject      s ' + ''.join([f'{i%10}' for i in range(len(unique_cells[:cell_area]))]) + '  ' + ''.join([c[0] for c in list(cov_lists.keys())[:cov_area]])
lines.append(header)
for idx,f in enumerate(files[:args.rows]):
    lines.append(render_row(idx, f, cov_lists, cell_map))

# footer with counts
lines.append('')
lines.append('Cell counts:')
for uc in unique_cells[:cell_area]:
    lines.append(f" {uc}: {sum(cell_map[uc])}")
if cov_lists:
    lines.append('\nCovariate summary:')
    for cname,vals in cov_lists.items():
        import statistics
        nums = [float(x) for x in vals if x is not None]
        if nums:
            lines.append(f" {cname}: n={len(nums)}, mean={statistics.mean(nums):.3f}, sd={statistics.pstdev(nums):.3f}")
        else:
            lines.append(f" {cname}: (no numeric values)")

out_path = Path(args.output) if args.output else p.parent / 'design_ascii.txt'
out_path.write_text('\n'.join(lines))
# Print top of file
print('\n'.join(lines[:min(len(lines), args.rows+5)]))
print(f'\nSaved ASCII preview to: {out_path}')
