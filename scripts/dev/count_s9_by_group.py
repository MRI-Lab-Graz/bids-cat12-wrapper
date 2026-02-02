import pandas as pd
import os
import glob


def count_subjects_with_data():
    participants_file = "participants.tsv"
    data_dir = "../data/cat12"

    # Read participants file
    try:
        df = pd.read_csv(participants_file, sep="\t")
    except Exception as e:
        print(f"Error reading {participants_file}: {e}")
        return

    # Ensure group column is treated as string/category to handle potential mixed types
    df["group_beh_factor"] = df["group_beh_factor"].astype(str)

    groups = sorted(df["group_beh_factor"].unique())

    print(f"{'Group':<10} | {'Total':<10} | {'With s9 Data':<15} | {'Missing':<10}")
    print("-" * 55)

    total_with_data = 0
    total_subjects = 0

    missing_subjects = []

    for group in groups:
        group_df = df[df["group_beh_factor"] == group]
        n_total = len(group_df)
        n_with_data = 0

        for _, row in group_df.iterrows():
            subject_id = row["participant_id"]

            # Check for s9 file (ses-1)
            # Pattern: s9mwp1*sub-XXXX_ses-1_acq-mprage_T1w.nii
            # We look in the mri subdirectory
            search_pattern = os.path.join(
                data_dir,
                subject_id,
                "mri",
                f"s9mwp1*{subject_id}_ses-1_acq-mprage_T1w.nii",
            )
            matches = glob.glob(search_pattern)

            if matches:
                n_with_data += 1
            else:
                missing_subjects.append((subject_id, group))

        print(
            f"{group:<10} | {n_total:<10} | {n_with_data:<15} | {n_total - n_with_data:<10}"
        )

        total_subjects += n_total
        total_with_data += n_with_data

    print("-" * 55)
    print(
        f"{'Total':<10} | {total_subjects:<10} | {total_with_data:<15} | {total_subjects - total_with_data:<10}"
    )

    if missing_subjects:
        print("\nSubjects missing s9 data (ses-1):")
        for sub, grp in missing_subjects:
            print(f"  {sub} (Group {grp})")


if __name__ == "__main__":
    count_subjects_with_data()
