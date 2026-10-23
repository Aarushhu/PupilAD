import argparse
import io
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from pupilad.paths import DS003838_RAW, DS003838_SLIM

BUCKET_URL = "https://s3.amazonaws.com/openneuro.org/ds003838"
SUBJECTS_WITHOUT_PUPIL = {"sub-017", "sub-094"}
DOWNLOAD_ATTEMPTS = 3
SLIM_COLUMNS = ["t", "eye", "conf", "diam", "blink"]
PUPIL_3D_ROWS_AWK = (
    'NR==1{next} $37=="pupil" && $8=="3d c++" ' '{print $1"\\t"$3"\\t"$4"\\t"$14"\\t"$36}'
)


def fetch_small_file(relative_path, destination_root=DS003838_RAW):
    destination = destination_root / relative_path
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-sfL", "-o", str(destination), f"{BUCKET_URL}/{relative_path}"],
        check=True,
    )
    return destination


def list_subjects():
    participants = pd.read_csv(fetch_small_file("participants.tsv"), sep="\t")
    return [
        subject for subject in participants.participant_id if subject not in SUBJECTS_WITHOUT_PUPIL
    ]


def stream_pupil_samples(subject):
    url = f"{BUCKET_URL}/{subject}/pupil/{subject}_task-memory_pupil.tsv"
    curl = subprocess.Popen(["curl", "-sfL", url], stdout=subprocess.PIPE)
    filtered = subprocess.run(
        ["awk", "-F\t", PUPIL_3D_ROWS_AWK],
        stdin=curl.stdout,
        capture_output=True,
        text=True,
        check=True,
    )
    curl.stdout.close()
    if curl.wait() != 0:
        raise RuntimeError(f"download failed for {subject}")
    return filtered.stdout


def parse_slim_samples(tsv_text):
    samples = pd.read_csv(io.StringIO(tsv_text), sep="\t", header=None, names=SLIM_COLUMNS)
    samples = samples.astype({"t": "float64", "eye": "int8", "conf": "float32", "diam": "float32"})
    samples["blink"] = pd.to_numeric(samples["blink"], errors="coerce").fillna(0).astype("int8")
    return samples


def download_subject_with_retries(subject):
    last_error = None
    for _ in range(DOWNLOAD_ATTEMPTS):
        try:
            return download_subject(subject)
        except (RuntimeError, subprocess.CalledProcessError) as error:
            last_error = error
    return subject, f"FAILED after {DOWNLOAD_ATTEMPTS} attempts: {last_error}"


def download_subject(subject):
    for relative_path in (
        f"{subject}/pupil/{subject}_task-memory_events.tsv",
        f"{subject}/beh/{subject}_task-memory_beh.tsv",
    ):
        try:
            fetch_small_file(relative_path)
        except subprocess.CalledProcessError:
            pass
    slim_path = DS003838_SLIM / f"{subject}.parquet"
    if slim_path.exists():
        return subject, "cached"
    slim_path.parent.mkdir(parents=True, exist_ok=True)
    samples = parse_slim_samples(stream_pupil_samples(subject))
    partial_path = slim_path.with_suffix(".partial")
    samples.to_parquet(partial_path, index=False)
    partial_path.rename(slim_path)
    return subject, f"{len(samples)} samples"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pupilad download")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args(argv)
    for name in ("participants.tsv", "participants.json", "dataset_description.json"):
        fetch_small_file(name)
    subjects = args.subjects or list_subjects()
    with ThreadPoolExecutor(args.workers) as pool:
        for subject, status in pool.map(download_subject_with_retries, subjects):
            print(subject, status, flush=True)


if __name__ == "__main__":
    main()
