#!/usr/bin/env python3
"""Upload results/*.jsonl to GCS after a bench run."""
import pathlib, sys
from google.cloud import storage

bucket_name, job_name = sys.argv[1], sys.argv[2]
b = storage.Client().bucket(bucket_name)
n = 0
for f in pathlib.Path("results").rglob("*"):
    if f.is_file():
        b.blob(f"{job_name}/{f.name}").upload_from_filename(str(f))
        print("uploaded", f.name)
        n += 1
print(f"{n} files -> gs://{bucket_name}/{job_name}/")
