"""transcribe-consensus: multi-model speech-to-text with an LLM-reconciled consensus transcript.

Global context: this is the orchestration wrapper. It reads config from `.env`/CLI, then runs
the phases in order: prepare audio chunks (audio.py) -> transcribe with each requested model
(stt.py) -> optionally merge into a consensus (merge.py) -> optionally proofread it for likely
mishearings (correct.py) -> write every intermediate result.
"""
import argparse
import difflib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .audio import prepare_chunks
from .correct import correct
from .log import get_logger
from .merge import consensus
from .stt import MODELS, Segment, Transcript, language_runs, transcribe

# gpt-transcribe for the best text, whisper-1 for timestamps, the diarize model for speaker labels.
DEFAULT_MODELS = "gpt-transcribe,whisper-1,gpt-4o-transcribe-diarize"
# OpenAI's "most capable model": https://developers.openai.com/api/docs/models/gpt-6-astra
DEFAULT_LLM = "gpt-6-astra"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI surface; every option can also be set in .env (see .env.example)."""
    p = argparse.ArgumentParser(prog="transcribe-consensus", description=__doc__.splitlines()[0])
    p.add_argument("input", type=Path, help="video or audio file")
    p.add_argument("--models", default=os.getenv("TRANSCRIBE_MODELS", DEFAULT_MODELS),
                   help=f"comma-separated subset of {list(MODELS)} (default: %(default)s)")
    p.add_argument("--languages", default=os.getenv("TRANSCRIBE_LANGUAGES", ""),
                   help="comma-separated ISO-639-1 codes; each language runs each model once (default: auto-detect)")
    p.add_argument("--prompt", default=os.getenv("TRANSCRIBE_PROMPT"), help="vocabulary/style hint passed to the STT models")
    p.add_argument("--context", default=os.getenv("TRANSCRIBE_CONTEXT"), help="description of the recording for the merge step")
    p.add_argument("--merge-model", default=os.getenv("TRANSCRIBE_MERGE_MODEL", DEFAULT_LLM), help="chat model for the consensus step")
    p.add_argument("--no-merge", action="store_true", help="skip the consensus step")
    p.add_argument("--correct-model", default=os.getenv("TRANSCRIBE_CORRECT_MODEL", DEFAULT_LLM), help="chat model for the error-correction step")
    p.add_argument("--no-correct", action="store_true", help="skip the error-correction step")
    p.add_argument("--out", type=Path, default=Path(os.getenv("TRANSCRIBE_OUT", "output")), help="output directory")
    p.add_argument("--from-raw", type=Path, help="skip audio+STT phases and re-run merge and correction on a previous raw.json")
    return p.parse_args(argv)


def load_raw(path: Path) -> list[Transcript]:
    """Inverse of the raw.json dump: rebuild Transcript objects so the merge phase can be re-run alone."""
    return [Transcript(d["run_id"], d["text"], [Segment(**s) for s in d["segments"]]) for d in json.loads(path.read_text(encoding="utf-8"))]


def write_outputs(out_dir: Path, stem: str, transcripts: list[Transcript], merged: str | None, corrected: str | None = None) -> Path:
    """Persist one .txt per model run, raw.json, consensus.txt, corrected.txt and the consensus->corrected diff."""
    d = out_dir / stem
    d.mkdir(parents=True, exist_ok=True)
    for t in transcripts:
        (d / f"{t.run_id}.txt").write_text(t.render(), encoding="utf-8")
    (d / "raw.json").write_text(json.dumps([asdict(t) for t in transcripts], ensure_ascii=False, indent=1), encoding="utf-8")
    if merged is not None:
        (d / "consensus.txt").write_text(merged, encoding="utf-8")
    if corrected is not None:
        (d / "corrected.txt").write_text(corrected, encoding="utf-8")
        before = merged if merged is not None else transcripts[0].render()  # what phase 4 was given
        # lineterm="" + "\n".join: difflib's documented way to diff inputs without trailing newlines
        diff = difflib.unified_diff(before.splitlines(), corrected.splitlines(), "input", "corrected.txt", n=0, lineterm="")
        (d / "corrections.diff").write_text("\n".join(diff) + "\n", encoding="utf-8")
    return d


def main(argv: list[str] | None = None) -> None:
    """Entry point: load .env, parse args, run the phases, print the final transcript."""
    load_dotenv()
    args = parse_args(argv)
    log = get_logger()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a legacy codepage
    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    specs = [MODELS[m.strip()] for m in args.models.split(",") if m.strip()]
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]  # empty = auto-detect

    if args.from_raw:
        log.info("phase 1-2/4: skipped, loading transcripts from %s", args.from_raw)
        transcripts = load_raw(args.from_raw)
    else:
        log.info("phase 1/4: preparing audio for %s", args.input)
        chunks = prepare_chunks(args.input, args.out / args.input.stem)
        log.info("phase 1/4: %d chunk(s)", len(chunks))
        log.info("phase 2/4: transcribing with %s, language hints %s", [s.name for s in specs], languages or "auto")
        transcripts = [transcribe(client, s, chunks, langs, args.prompt) for s in specs for langs in language_runs(s, languages)]

    merged = None
    if not args.no_merge and len(transcripts) > 1:
        log.info("phase 3/4: merging %d transcripts with %s", len(transcripts), args.merge_model)
        merged = consensus(client, transcripts, args.merge_model, args.context)
    else:
        log.info("phase 3/4: merge skipped")
    best = merged if merged is not None else transcripts[0].render()

    corrected = None
    if not args.no_correct:
        log.info("phase 4/4: correcting likely mistranscriptions with %s", args.correct_model)
        corrected = correct(client, best, transcripts, args.correct_model, args.context)
    else:
        log.info("phase 4/4: correction skipped")

    d = write_outputs(args.out, args.input.stem, transcripts, merged, corrected)
    log.info("done: outputs in %s", d)
    print(corrected if corrected is not None else best)
