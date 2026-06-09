# Data Handling

This repository does not store raw, merged, or prepared imagery.

Point the code at the cleaned merged dataset with:

```bash
export FISHDETECT_DATASET_ROOT=/path/to/cleaned_merged_viame_dive_dataset
```

Prepared training artifacts should also live outside the repository:

```bash
export FISHDETECT_PREPARED_ROOT=/path/to/prepared_outputs
```

Preparation scripts create symlinks to the merged image files where possible. They never edit or overwrite the master merged dataset.
