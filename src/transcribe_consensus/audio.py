"""Phase 1: turn any video/audio file into small mono mp3 chunks the API accepts.

Global context: OpenAI's audio endpoints reject uploads over 25 MB
(https://platform.openai.com/docs/guides/speech-to-text#longer-inputs). We downmix to mono
16 kHz 64 kbps mp3 (plenty for speech) and, only if that is still too big, split it into
time-based segments with ffmpeg. Later phases transcribe each chunk and shift timestamps.
"""
import subprocess
from pathlib import Path

MAX_BYTES = 25 * 1024 * 1024  # OpenAI hard limit per upload


def _run(cmd: list[str]) -> str:
    """Run a command, raise on failure, return stdout (helper so every ffmpeg call is one line)."""
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def duration_seconds(path: Path) -> float:
    """Media duration via ffprobe. Source: https://trac.ffmpeg.org/wiki/FFprobeTips#Duration"""
    return float(_run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]))


def extract_audio(src: Path, dst: Path) -> Path:
    """Write a mono 16 kHz 64 kbps mp3 of `src` to `dst`.

    -vn drops video, -ac 1 mono, -ar 16000 speech sample rate, -b:a 64k small bitrate.
    Source: https://ffmpeg.org/ffmpeg.html#Audio-Options
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", str(dst)])
    return dst


def chunk_seconds(total_seconds: float, size_bytes: int, limit: int = MAX_BYTES) -> int:
    """Segment length that keeps each chunk under `limit`, with ~10% headroom for container overhead."""
    if size_bytes <= limit:
        return 0  # 0 = no splitting needed
    return max(60, int(total_seconds * limit * 0.9 / size_bytes))  # never shorter than a minute


def split_audio(mp3: Path, seconds: int) -> list[Path]:
    """Split `mp3` into `seconds`-long files with ffmpeg's segment muxer (stream copy, no re-encode).

    Source: https://ffmpeg.org/ffmpeg-formats.html#segment_002c-stream_005fsegment_002c-ssegment
    """
    pattern = mp3.with_name(mp3.stem + ".part%03d.mp3")
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(mp3), "-f", "segment", "-segment_time", str(seconds), "-c", "copy", str(pattern)])
    return sorted(mp3.parent.glob(mp3.stem + ".part*.mp3"))


def prepare_chunks(src: Path, work_dir: Path) -> list[tuple[Path, float]]:
    """Full phase: extract audio, split if needed, return [(chunk_path, start_offset_seconds), ...].

    Offsets come from measured chunk durations so timestamps stay accurate after concatenation.
    """
    mp3 = extract_audio(src, work_dir / (src.stem + ".mp3"))
    secs = chunk_seconds(duration_seconds(mp3), mp3.stat().st_size)
    parts = split_audio(mp3, secs) if secs else [mp3]
    offsets, t = [], 0.0
    for p in parts:  # cumulative start time of each part
        offsets.append((p, t))
        t += duration_seconds(p)
    return offsets
