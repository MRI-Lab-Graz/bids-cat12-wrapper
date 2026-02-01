from nilearn import datasets
try:
    aal = datasets.fetch_atlas_aal()
    print("AAL atlas fetched successfully.")
    print(aal.maps)
    print(aal.labels[:5])
except Exception as e:
    print(f"Error fetching atlas: {e}")
