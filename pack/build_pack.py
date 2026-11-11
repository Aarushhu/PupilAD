import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

PYTHON_VERSION = "3.12.10"
PYTHON_TAG = "312"
EMBED_URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"
RELEASE_TAG = "pack-latest"
ZIP_NAME = "PupilAD-Pack.zip"

REPO = Path(__file__).resolve().parents[1]
PACK_FILES = Path(__file__).resolve().parent / "files"
REQUIREMENTS = REPO / "requirements-app.txt"


def log(message):
    print(f"[pack] {message}", flush=True)


def run(*command, cwd=REPO):
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def requirements_key():
    text = REQUIREMENTS.read_text(encoding="utf-8")
    return hashlib.sha256(f"{PYTHON_VERSION}\n{text}".encode()).hexdigest()


def install_python(runtime):
    log(f"Downloading Python {PYTHON_VERSION} (embeddable)")
    data = urllib.request.urlopen(EMBED_URL, timeout=120).read()
    if runtime.exists():
        shutil.rmtree(runtime)
    runtime.mkdir(parents=True)
    zipfile.ZipFile(io.BytesIO(data)).extractall(runtime)
    (runtime / f"python{PYTHON_TAG}._pth").write_text(
        f"python{PYTHON_TAG}.zip\n.\nLib\\site-packages\n..\\..\\src\nimport site\n", encoding="utf-8"
    )


def install_packages(runtime):
    target = runtime / "Lib" / "site-packages"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    log("Installing packages from requirements-app.txt")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
            "--target", str(target), "--platform", "win_amd64", "--python-version", PYTHON_TAG,
            "--implementation", "cp", "--only-binary=:all:", "-r", str(REQUIREMENTS),
        ],
        check=True,
    )
    for folder in target.rglob("__pycache__"):
        shutil.rmtree(folder, ignore_errors=True)


def ensure_runtime(pack, rebuild):
    runtime = pack / "runtime" / "python"
    marker = runtime / "pack-requirements.sha256"
    key = requirements_key()
    if rebuild or not (runtime / "python.exe").exists():
        install_python(runtime)
        install_packages(runtime)
    elif not marker.exists() or marker.read_text().strip() != key:
        install_packages(runtime)
    else:
        log("Python and packages are up to date")
        return
    marker.write_text(key, encoding="utf-8")


def sync_tree(source, target):
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(source, target, ignore=ignore, dirs_exist_ok=True)
    wanted = {
        path.relative_to(source)
        for path in source.rglob("*")
        if "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    for path in sorted(target.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.relative_to(target) in wanted or "__pycache__" in path.parts:
            continue
        try:
            path.rmdir() if path.is_dir() else path.unlink()
        except OSError:
            log(f"Could not remove old {path}; it is unused and can be deleted by hand")


def copy_code(pack):
    sync_tree(REPO / "src" / "pupilad", pack / "src" / "pupilad")
    for item in PACK_FILES.iterdir():
        shutil.copy2(item, pack / item.name)
    (pack / "data").mkdir(exist_ok=True)


def write_version(pack):
    commit = run("git", "rev-parse", "--short", "HEAD")
    dirty = bool(run("git", "status", "--porcelain", "--", "src", "requirements-app.txt", "pack"))
    info = {
        "commit": commit + ("+local-changes" if dirty else ""),
        "built": datetime.now().isoformat(timespec="seconds"),
        "python": PYTHON_VERSION,
    }
    (pack / "VERSION.txt").write_text(
        f"PupilAD pack\ncode version: {info['commit']}\nbuilt: {info['built']}\npython: {info['python']}\n",
        encoding="utf-8",
    )
    return info


def smoke_test(pack):
    python = pack / "runtime" / "python" / "python.exe"
    script = (
        "import cv2, numpy as np, flask, pandas, cryptography;"
        "from pupilad.app.server import create_app;"
        "from pupilad.app.checks import inspect_frame;"
        "ok, jpg = cv2.imencode('.jpg', np.full((240, 320, 3), 128, np.uint8));"
        "assert inspect_frame(jpg.tobytes())['faces'] == 0;"
        "import tempfile; app = create_app(tempfile.mkdtemp(), auto_process=False);"
        "assert app.test_client().get('/api/info').status_code == 200;"
        "print('ok', cv2.__version__, np.__version__, pandas.__version__, flask.__version__)"
    )
    output = subprocess.run([str(python), "-c", script], check=True, capture_output=True, text=True).stdout
    log(f"Smoke test passed: {output.strip()}")


def make_zip(pack, destination):
    log(f"Zipping to {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(pack.rglob("*")):
            relative = path.relative_to(pack)
            if relative.parts[0] == "data" or "__pycache__" in relative.parts:
                continue
            if path.is_file():
                archive.write(path, Path(pack.name) / relative)
    log(f"Zip size: {destination.stat().st_size / 1e6:.0f} MB")


def publish(zip_path, info):
    notes = (
        f"Self-contained PupilAD app for Windows (64-bit). Code version {info['commit']}, "
        f"built {info['built']}, Python {info['python']}.\n\n"
        "Unzip onto a flash drive or any folder, then double-click `Start PupilAD.bat`. "
        "No Python or internet needed. See README.txt inside."
    )
    subprocess.run(["gh", "release", "delete", RELEASE_TAG, "--yes", "--cleanup-tag"], cwd=REPO, capture_output=True)
    run("git", "fetch", "--tags", "--prune", "--prune-tags", "--quiet")
    run(
        "gh", "release", "create", RELEASE_TAG, str(zip_path), "--title", "PupilAD Pack (latest)",
        "--notes", notes, "--target", run("git", "rev-parse", "HEAD"),
    )
    log(f"Published release {RELEASE_TAG}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the self-contained PupilAD pack for Windows")
    parser.add_argument("--out", type=Path, default=REPO.parent / "PupilAD-Pack")
    parser.add_argument("--rebuild", action="store_true", help="download Python and packages again")
    parser.add_argument("--zip", action="store_true", help=f"also write {ZIP_NAME} next to the pack")
    parser.add_argument("--publish", action="store_true", help="zip and upload to the GitHub release")
    args = parser.parse_args(argv)
    pack = args.out.resolve()
    pack.mkdir(parents=True, exist_ok=True)
    log(f"Building {pack}")
    ensure_runtime(pack, args.rebuild)
    copy_code(pack)
    info = write_version(pack)
    smoke_test(pack)
    if args.zip or args.publish:
        zip_path = pack.parent / ZIP_NAME
        make_zip(pack, zip_path)
        if args.publish:
            publish(zip_path, info)
    log(f"Done. Code version {info['commit']}")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
