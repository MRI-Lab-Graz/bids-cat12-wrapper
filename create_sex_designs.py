#!/usr/bin/env python3
import json
import pandas as pd

# Load full design
with open('results/vbm/3groups_vbm_smooth9_age_sex_tiv/design.json') as f:
    design = json.load(f)

# Load participant lists
df_f = pd.read_csv('participants_females_only.tsv', sep='\t')
df_m = pd.read_csv('participants_males_only.tsv', sep='\t')

female_ids = set(str(pid) for pid in df_f['participant_id'])
male_ids = set(str(pid) for pid in df_m['participant_id'])

# Filter design for females
design_f = {
    'modality': design['modality'],
    'smoothing': design['smoothing'],
    'groups': {},
    'files': []
}

for group, group_data in design['groups'].items():
    design_f['groups'][group] = {'sessions': {}}
    for session, files in group_data['sessions'].items():
        filtered = [f for f in files if any(sid in f for sid in female_ids)]
        if filtered:
            design_f['groups'][group]['sessions'][session] = filtered

# Build files list from design files
for entry in design.get('files', []):
    if entry['subject'] in female_ids:
        design_f['files'].append(entry)

# Filter design for males
design_m = {
    'modality': design['modality'],
    'smoothing': design['smoothing'],
    'groups': {},
    'files': []
}

for group, group_data in design['groups'].items():
    design_m['groups'][group] = {'sessions': {}}
    for session, files in group_data['sessions'].items():
        filtered = [f for f in files if any(sid in f for sid in male_ids)]
        if filtered:
            design_m['groups'][group]['sessions'][session] = filtered

# Build files list from design files
for entry in design.get('files', []):
    if entry['subject'] in male_ids:
        design_m['files'].append(entry)

# Save
with open('results/vbm/3groups_vbm_smooth9_age_sex_tiv/design_females.json', 'w') as f:
    json.dump(design_f, f, indent=2)

with open('results/vbm/3groups_vbm_smooth9_age_sex_tiv/design_males.json', 'w') as f:
    json.dump(design_m, f, indent=2)

# Count subjects
f_count = sum(len(design_f['groups'][g]['sessions'].get('1', [])) for g in design_f['groups'])
m_count = sum(len(design_m['groups'][g]['sessions'].get('1', [])) for g in design_m['groups'])

print(f'Females design: {f_count} subjects')
print(f'Males design: {m_count} subjects')
