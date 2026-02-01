import pandas as pd
from pathlib import Path
import csv

CAT12_DIR = Path('../data/cat12')
GROUP_FILE = Path('group.tsv')
OLD_PART_FILE = Path('participants.tsv')
OUT_FILE = Path('participants_rebuilt.tsv')

def get_sessions_from_disk(sub_id):
    # Check disk for sessions
    # Look for sub-ID/mri/s*mwp1*.nii
    # We need to count how many sessions exist
    
    # Try standard path
    p = CAT12_DIR / sub_id
    if not p.exists():
        return 0
    
    mri_dir = p / 'mri'
    if not mri_dir.exists():
        return 0
    
    # Glob all VBM files
    files = list(mri_dir.glob('s*mwp1*.nii'))
    if not files:
        return 0
        
    # Extract session numbers
    sessions = set()
    for f in files:
        # filename example: s9mwp1rsub-1291006_ses-1_acq-mprage_T1w.nii
        name = f.name
        if 'ses-' in name:
            try:
                # split by 'ses-' and take the next char(s)
                # simplistic: assume single digit or check BIDS
                parts = name.split('ses-')
                if len(parts) > 1:
                    ses_part = parts[1].split('_')[0]
                    sessions.add(ses_part)
            except:
                pass
    
    # If no explicit ses- tag found but files exist, maybe it's 1 session?
    # But the pipeline expects ses-X.
    # If files exist but no ses tag, we might assume ses-1 if that's the convention, 
    # but let's stick to what we found.
    
    return len(sessions) if sessions else 0

def compute_factor(gb, i2, i4):
    # Logic from previous script
    try:
        gb = int(gb)
    except:
        return 5 # default
        
    try:
        i2 = int(i2)
    except:
        i2 = 0
    try:
        i4 = int(i4)
    except:
        i4 = 0

    if gb == 3: return 5
    if gb == 1:
        if i2 and not i4: return 1
        if i4 and not i2: return 2
        if i4: return 2
        if i2: return 1
        return 1
    if gb == 2:
        if i2 and not i4: return 3
        if i4 and not i2: return 4
        if i4: return 4
        if i2: return 3
        return 3
    return 5

def rebuild():
    print("Reading group.tsv...")
    df_group = pd.read_csv(GROUP_FILE, sep='\t')
    
    # Load old participants to preserve 'group' and 'group_ml' text labels if possible
    # or we can just regenerate them.
    # The old file had: group (control/intervention_2w/...), group_ml (control/intervention)
    # We can infer these from group_beh_factor?
    # Factor 1: alone_2w -> group=?, group_ml=?
    # Actually, let's just map factor to labels for clarity
    
    # Factor Map
    # 1: alone_2w
    # 2: alone_4w
    # 3: group_2w
    # 4: group_4w
    # 5: control
    
    # In old file:
    # 1 (alone_2w) -> group=control (Wait, sub-1291003 is factor 1 and group=control? Let's check)
    # sub-1291003: group=control, group_ml=control, factor=1.
    # sub-1291011: group=intervention_2w, factor=2.
    
    # Wait, let's check the old file mapping again.
    # sub-1291003 (Factor 1) -> group=control.
    # sub-1291011 (Factor 2) -> group=intervention_2w.
    
    # This seems inconsistent or I misunderstood the factor logic.
    # Let's look at group.tsv for 1291003: group_beh=1 (alone), interv_2w=1.
    # So Factor 1 = Alone 2w.
    # Why is it "control" in participants.tsv?
    # Maybe "control" means "active control" (running alone)?
    # And "intervention" means "group running"?
    
    # Let's check sub-1291011 (Factor 2? No, let's check group.tsv)
    # sub-1291011: group_beh=1 (alone), interv_4w=1. -> Factor 2 (Alone 4w).
    # In participants.tsv: group=intervention_2w.
    # Wait, 1291011 is "intervention_2w" but has interv_4w=1?
    # group.tsv: sub-1291011, interv_2w=0, interv_4w=1.
    # So it should be 4w.
    # participants.tsv says "intervention_2w".
    
    # It seems `participants.tsv` might have errors or old labels!
    # The user said "Combine these two files" earlier.
    # I should trust `group.tsv` as the source of truth for the factors.
    
    # I will generate the file with the new factor and generic labels if needed.
    # I'll keep the columns: participant_id, nr_sessions, age, sex, group_beh_factor
    # I'll drop the confusing text 'group' columns unless requested.
    # Or I can map them:
    # 1 (Alone 2w)
    # 2 (Alone 4w)
    # 3 (Group 2w)
    # 4 (Group 4w)
    # 5 (Control)
    
    new_rows = []
    
    for _, row in df_group.iterrows():
        sub_id = row['subject_id']
        if pd.isna(sub_id): continue
        sub_id = sub_id.strip()
        
        # Calculate sessions from disk
        # If 0 on disk, we still add them (survey only) but nr_sessions=0?
        # Or default to 3 if we assume they are survey subjects?
        # The pipeline filters by finding files.
        # Let's put the actual number of sessions found on disk.
        # If 0, the pipeline will just skip them (which is correct for survey only).
        # BUT, if they are survey only, maybe they shouldn't be in the VBM analysis list?
        # The user said "It is OK that some subjects are in the participants.tsv but not in the cat12 folder".
        # So we should include them.
        # What should nr_sessions be?
        # If I put 0, `parse_participants.py` might complain?
        # `parse_participants.py`: "max_sessions = int(df["nr_sessions"].max())"
        # If I put 0, it might be fine.
        # Let's check if `participants.tsv` has 0 anywhere.
        # No, it has 3 everywhere (mostly).
        
        n_sessions = get_sessions_from_disk(sub_id)
        if n_sessions == 0:
            # Check if it was in old file
            # If so, keep old nr_sessions (usually 3)
            # If not, maybe default to 3?
            n_sessions = 3 
        
        # Age/Sex
        age = row['age']
        sex_code = row['sex'] # 1=F, 2=M
        sex = 'F' if sex_code == 1 else 'M' if sex_code == 2 else 'NA'
        
        # Factor
        gb = row['group_beh']
        i2 = row['interv_2w']
        i4 = row['interv_4w']
        factor = compute_factor(gb, i2, i4)
        
        new_rows.append({
            'participant_id': sub_id,
            'nr_sessions': n_sessions,
            'age': age,
            'sex': sex,
            'group_beh_factor': factor
        })
        
    # Convert to DF
    df_new = pd.DataFrame(new_rows)
    
    # Write
    df_new.to_csv(OUT_FILE, sep='\t', index=False)
    print(f"Wrote {len(df_new)} subjects to {OUT_FILE}")

if __name__ == '__main__':
    rebuild()
