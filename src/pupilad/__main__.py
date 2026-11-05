import importlib
import sys

COMMANDS = {
    "app": "pupilad.app.server",
    "download": "pupilad.ds003838.download",
    "preprocess": "pupilad.ds003838.preprocess",
    "features": "pupilad.ds003838.features",
    "validate": "pupilad.ds003838.validate",
    "train": "pupilad.validation.synthetic_recovery",
    "lc-check": "pupilad.validation.lc_score",
    "tobii-features": "pupilad.tobii.ingest",
    "plr-features": "pupilad.plr.ingest",
    "extract-webcam": "pupilad.webcam.video_extraction",
    "webcam-trials": "pupilad.webcam.trial_table",
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(f"usage: pupilad {{{','.join(COMMANDS)}}} [options]")
        return 2
    importlib.import_module(COMMANDS[argv[0]]).main(argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
