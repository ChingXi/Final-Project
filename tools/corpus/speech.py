
import os
import subprocess
import sys

VOICE_NOTES = [
    (
        "steering_update_march.wav",
        "Microsoft David Desktop",
        "Steering committee update, March twenty-fourth. On Project Atlas, "
        "we're moving the launch from August to the thirtieth of September "
        "because of the vendor delay, and the remaining budget is being "
        "reallocated to additional quality assurance testing. David Okafor "
        "is taking over as Atlas owner from Maria, effective next week.",
    ),
    (
        "delivery_note_march.wav",
        "Microsoft Zira Desktop",
        "Quick note on delivery. Borealis has been brought forward and is now "
        "ahead of its original date. Separately, the authentication migration "
        "on Fathom still has no rollback plan, which we flagged as the main "
        "risk in the meeting.",
    ),
    (
        "vendor_note_march.wav",
        "Microsoft David Desktop",
        "Note on vendors. The analytics vendor contract is still unsigned, and "
        "procurement have asked for a decision before the end of the quarter. "
        "No other supplier changes to report this month.",
    ),
]

RATES = (170, 170, 145)

MIN_WAV_BYTES = 50_000


def _reference_name(filename: str) -> str:
    return filename.replace(".wav", ".txt")


def write_references(out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for filename, _voice, script in VOICE_NOTES:
        path = os.path.join(out_dir, _reference_name(filename))
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)
        written.append(path)
    return written


def render_one(index: int, out_dir: str) -> str:
    import pyttsx3

    filename, voice_hint, script = VOICE_NOTES[index]
    engine = pyttsx3.init()
    available = {v.name: v.id for v in engine.getProperty("voices")}
    match = next(
        (vid for name, vid in available.items()
         if voice_hint.lower() in name.lower()),
        None,
    )
    if match is None:
        print(f"  voice {voice_hint!r} unavailable, using the default; "
              f"installed: {sorted(available)}")
    else:
        engine.setProperty("voice", match)
    engine.setProperty("rate", RATES[index])

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    engine.save_to_file(script, path)
    engine.runAndWait()
    return path


def synthesise(out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    written = []
    for index, (filename, _voice, _script) in enumerate(VOICE_NOTES):
        path = os.path.join(out_dir, filename)
        if os.path.exists(path):
            os.remove(path)
        subprocess.run(
            [sys.executable, "-m", "tools.corpus.speech", str(index),
             os.path.abspath(out_dir)],
            cwd=root, check=True, timeout=120,
        )

        size = os.path.getsize(path) if os.path.exists(path) else 0
        if size < MIN_WAV_BYTES:
            raise RuntimeError(
                f"{path} is {size} bytes, which is a WAV header and no audio. "
                f"SAPI produced no speech for {filename}."
            )
        written.append(path)

    write_references(out_dir)
    return written


if __name__ == "__main__":
    render_one(int(sys.argv[1]), sys.argv[2])
