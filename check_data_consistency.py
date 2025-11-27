import pandas as pd
from pathlib import Path
import sys

# Config
CAT12_DIR = Path('../data/cat12')
PARTICIPANTS_FILE = Path('participants.tsv')

def check_consistency():
    if not PARTICIPANTS_FILE.exists():
        print("participants.tsv not found")
        return

    # 1. Load Participants List
    df = pd.read_csv(PARTICIPANTS_FILE, sep='\t')
    tsv_subjects = set(df['participant_id'].unique())
    print(f"Subjects in participants.tsv: {len(tsv_subjects)}")

    # 2. Scan Disk for Subject Folders
    disk_subjects = set()
    r_files = 0
    no_r_files = 0
    
    if CAT12_DIR.exists():
        for p in CAT12_DIR.glob('sub-*'):
            if p.is_dir():
                sub_id = p.name
                disk_subjects.add(sub_id)
                
                # Check file naming convention for this subject
                # Look for s9mwp1*
                mri_dir = p / 'mri'
                if mri_dir.exists():
                    # Check for 'r'
                    if list(mri_dir.glob('s9mwp1r*.nii')):
                        r_files += 1
                    elif list(mri_dir.glob('s9mwp1*.nii')):
                        no_r_files += 1
    
    print(f"Subjects found on disk:       {len(disk_subjects)}")
    print(f"  - With 'r' (s9mwp1r...):    {r_files}")
    print(f"  - No 'r'   (s9mwp1...):     {no_r_files}")

    # 3. Compare
    missing_from_disk = sorted(list(tsv_subjects - disk_subjects))
    missing_from_tsv = sorted(list(disk_subjects - tsv_subjects))

    print("\n--- DISCREPANCY REPORT ---")
    print(f"1. In TSV but MISSING from disk: {len(missing_from_disk)}")
    if missing_from_disk:
        print(f"   Examples: {', '.join(missing_from_disk[:5])} ...")

    print(f"\n2. On disk but MISSING from TSV: {len(missing_from_tsv)}")
    if missing_from_tsv:
        print(f"   Examples: {', '.join(missing_from_tsv[:5])} ...")

if __name__ == '__main__':
    check_consistency()
