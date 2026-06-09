# Data And Privacy Notes

This repository is intended to be public-facing. It should contain code, tests, documentation, and small synthetic fixtures only.

Do not commit:

```text
raw images
merged images
prepared training exports
model weights
prediction galleries
local output folders
absolute filesystem paths
restricted source manifests
non-public project notes
```

The real dataset is referenced through environment variables:

```bash
export FISHDETECT_DATASET_ROOT=/path/to/cleaned_dataset
export FISHDETECT_PREPARED_ROOT=/path/to/prepared_outputs
```

Prepared metadata strips source provenance down to public-safe basenames. Generated outputs are ignored by git by default.

Before publishing a release, run:

```bash
rg -n "/Users/|/home/|source_path|personal|restricted" .
find . -size +50M -type f
```

Review any matches before pushing.
