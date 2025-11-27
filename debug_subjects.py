import pandas as pd
from pathlib import Path

CAT12_DIR = Path('../data/cat12')
PARTICIPANTS_FILE = Path('participants.tsv')

def analyze_subjects():
    # 1. Load TSV
    if not PARTICIPANTS_FILE.exists():
        print("No participants.tsv")
        return
    df = pd.read_csv(PARTICIPANTS_FILE, sep='\t')
    tsv_subjects = set(df['participant_id'].unique())

    # 2. Scan Disk
    disk_subjects = set()
    missing_r_subjects = set()
    
    if CAT12_DIR.exists():
        for p in CAT12_DIR.glob('sub-*'):
            if p.is_dir():
                sub_id = p.name
                disk_subjects.add(sub_id)
                
                # Check for missing 'r'
                mri_dir = p / 'mri'
                if mri_dir.exists():
                    # If we have s9mwp1... but NO s9mwp1r...
                    has_r = list(mri_dir.glob('s9mwp1r*.nii'))
                    has_no_r = list(mri_dir.glob('s9mwp1sub*.nii'))
                    
                    if has_no_r and not has_r:
                        missing_r_subjects.add(sub_id)

    # 3. Analysis
    on_disk_not_in_tsv = sorted(list(disk_subjects - tsv_subjects))
    in_tsv_not_on_disk = sorted(list(tsv_subjects - disk_subjects))
    
    # Intersection (subjects that SHOULD be in analysis)
    valid_subjects = disk_subjects.intersection(tsv_subjects)
    
    # Check if "missing r" subjects are in the valid set
    missing_r_in_tsv = missing_r_subjects.intersection(tsv_subjects)
    missing_r_not_in_tsv = missing_r_subjects - tsv_subjects

    print(f"Total subjects in TSV: {len(tsv_subjects)}")
    print(f"Total subjects on Disk: {len(disk_subjects)}")
    print(f"Subjects in both (Expected Analysis N): {len(valid_subjects)}")
    
    print("\n--- POTENTIAL DATA LOSS (On Disk but NOT in TSV) ---")
    print(f"Count: {len(on_disk_not_in_tsv)}")
    print(f"IDs: {', '.join(on_disk_not_in_tsv)}")
    
    print("\n--- FILENAME IRREGULARITY (Missing 'r' in filename) ---")
    print(f"Count: {len(missing_r_subjects)}")
    print(f"IDs: {', '.join(sorted(list(missing_r_subjects)))}")
    
    print("\n--- OVERLAP CHECK ---")
    print(f"Subjects with missing 'r' that are IN the TSV: {len(missing_r_in_tsv)}")
    if missing_r_in_tsv:
        print(f"IDs: {', '.join(sorted(list(missing_r_in_tsv)))}")
        
    print(f"Subjects with missing 'r' that are NOT in the TSV: {len(missing_r_not_in_tsv)}")

if __name__ == '__main__':
    analyze_subjects()
