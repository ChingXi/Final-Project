
import os
import re
import sys

import jiwer

from akg.config import load_settings
from akg.ingest.audio import transcribe

AUDIO_DIR = os.path.join("data", "audio")
RECORDED_DIR = os.path.join("eval", "recordings")
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}

TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])


NUMERALS = {
    "twenty-fourth": "24th",
    "twenty fourth": "24th",
    "thirtieth": "30th",
    "fifteenth": "15th",
    "tenth": "10th",
}


def normalise(text: str) -> str:
    lowered = text.lower()
    for word, digits in NUMERALS.items():
        lowered = lowered.replace(word, digits)
    return TRANSFORM(lowered)


def reference_for(audio_path: str) -> str:
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    return os.path.join(AUDIO_DIR, f"{stem.replace('_recorded', '')}.txt")


def measure(settings, audio_path: str, reference_path: str | None = None) -> dict:
    reference_path = reference_path or reference_for(audio_path)
    if not os.path.exists(reference_path):
        raise FileNotFoundError(
            f"{audio_path} has no reference transcript at {reference_path}; "
            f"word error rate cannot be measured against a script that was "
            f"not kept"
        )
    with open(reference_path, encoding="utf-8") as f:
        reference = f.read()

    chunks = transcribe(audio_path, settings, doc_hash="wer")
    hypothesis = " ".join(c.text for c in chunks)

    # jiwer 4 removed compute_measures in favour of process_words.
    out = jiwer.process_words(TRANSFORM(reference), TRANSFORM(hypothesis))
    return {
        "file": os.path.basename(audio_path),
        "words": len(TRANSFORM(reference).split()),
        "wer": out.wer,
        "wer_norm": jiwer.wer(normalise(reference), normalise(hypothesis)),
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "hypothesis": hypothesis.strip(),
    }


def _audio_files(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if os.path.splitext(name)[1].lower() in AUDIO_EXTS
    )


def _weighted(rows: list[dict], key: str) -> float | None:
    total = sum(r["words"] for r in rows)
    if not total:
        return None
    return sum(r[key] * r["words"] for r in rows) / total


def report(settings) -> dict:
    out: dict = {}
    for group, directory in (
        ("synthetic", AUDIO_DIR),
        ("recorded", RECORDED_DIR),
    ):
        rows = [measure(settings, path) for path in _audio_files(directory)]
        out[group] = {
            "notes": rows,
            "wer": _weighted(rows, "wer"),
            "wer_norm": _weighted(rows, "wer_norm"),
        }
    return out


def _print_group(name: str, group: dict) -> None:
    if not group["notes"]:
        print(f"{name}: no audio found")
        return
    print(f"{name}:")
    for row in group["notes"]:
        print(f"  {row['file']:<40} {row['wer'] * 100:>6.2f}%  "
              f"norm {row['wer_norm'] * 100:>6.2f}%  "
              f"({row['substitutions']}S {row['deletions']}D "
              f"{row['insertions']}I over {row['words']} words)")
    print(f"  {'weighted':<40} {group['wer'] * 100:>6.2f}%  "
          f"norm {group['wer_norm'] * 100:>6.2f}%")


def main() -> None:
    settings = load_settings()

    if len(sys.argv) == 3:
        row = measure(settings, sys.argv[1], sys.argv[2])
        print(f"audio      : {sys.argv[1]}")
        print(f"reference  : {sys.argv[2]}")
        print(f"ASR model  : {settings.asr_model} ({settings.asr_device})\n")
        print(f"WER        : {row['wer'] * 100:.2f}%")
        print(f"  normalised   : {row['wer_norm'] * 100:.2f}%")
        print(f"  substitutions: {row['substitutions']}")
        print(f"  deletions    : {row['deletions']}")
        print(f"  insertions   : {row['insertions']}\n")
        print("hypothesis :", row["hypothesis"])
        return

    if len(sys.argv) != 1:
        sys.exit("usage: python eval/wer.py [<audio-file> <reference-file>]")

    print(f"ASR model  : {settings.asr_model} ({settings.asr_device})\n")
    out = report(settings)
    _print_group("synthetic", out["synthetic"])
    print()
    _print_group("recorded", out["recorded"])


if __name__ == "__main__":
    main()
