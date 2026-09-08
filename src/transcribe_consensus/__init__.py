"""transcribe-consensus: multi-model speech-to-text with an LLM-reconciled consensus transcript.

Global context: this is the orchestration wrapper. It reads config from `.env`/CLI, then runs
the phases in order: prepare audio chunks (audio.py) -> transcribe with each requested model
(stt.py) -> write raw outputs -> optionally merge into a consensus (merge.py).
"""
import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .audio import prepare_chunks
from .log import get_logger
from .merge import consensus
from .stt import MODELS, Segment, Transcript, transcribe

DEFAULT_MODELS = "whisper-1,gpt-4o-transcribe,gpt-4o-transcribe-diarize"


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
    p.add_argument("--merge-model", default=os.getenv("TRANSCRIBE_MERGE_MODEL", "gpt-5"), help="chat model for the consensus step")
    p.add_argument("--no-merge", action="store_true", help="skip the consensus step")
    p.add_argument("--out", type=Path, default=Path(os.getenv("TRANSCRIBE_OUT", "output")), help="output directory")
    p.add_argument("--from-raw", type=Path, help="skip audio+STT phases and re-run only the merge on a previous raw.json")
    return p.parse_args(argv)


def load_raw(path: Path) -> list[Transcript]:
    """Inverse of the raw.json dump: rebuild Transcript objects so the merge phase can be re-run alone."""
    return [Transcript(d["run_id"], d["text"], [Segment(**s) for s in d["segments"]]) for d in json.loads(path.read_text(encoding="utf-8"))]


def write_outputs(out_dir: Path, stem: str, transcripts: list[Transcript], merged: str | None) -> Path:
    """Persist one .txt per model run, one raw .json with everything, and the consensus .txt."""
    d = out_dir / stem
    d.mkdir(parents=True, exist_ok=True)
    for t in transcripts:
        (d / f"{t.run_id}.txt").write_text(t.render(), encoding="utf-8")
    (d / "raw.json").write_text(json.dumps([asdict(t) for t in transcripts], ensure_ascii=False, indent=1), encoding="utf-8")
    if merged is not None:
        (d / "consensus.txt").write_text(merged, encoding="utf-8")
    return d


def main(argv: list[str] | None = None) -> None:
    """Entry point: load .env, parse args, run the phases, print the final transcript."""
    load_dotenv()
    args = parse_args(argv)
    log = get_logger()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a legacy codepage
    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    specs = [MODELS[m.strip()] for m in args.models.split(",") if m.strip()]
    languages = [l.strip() for l in args.languages.split(",") if l.strip()] or [None]

    if args.from_raw:
        log.info("phase 1-2/3: skipped, loading transcripts from %s", args.from_raw)
        transcripts = load_raw(args.from_raw)
    else:
        log.info("phase 1/3: preparing audio for %s", args.input)
        chunks = prepare_chunks(args.input, args.out / args.input.stem)
        log.info("phase 1/3: %d chunk(s)", len(chunks))
        log.info("phase 2/3: transcribing with %s x languages %s", [s.name for s in specs], languages)
        transcripts = [transcribe(client, s, chunks, lang, args.prompt) for s in specs for lang in languages]

    merged = None
    if not args.no_merge and len(transcripts) > 1:
        log.info("phase 3/3: merging %d transcripts with %s", len(transcripts), args.merge_model)
        merged = consensus(client, transcripts, args.merge_model, args.context)
    else:
        log.info("phase 3/3: merge skipped")

    d = write_outputs(args.out, args.input.stem, transcripts, merged)
    log.info("done: outputs in %s", d)
    print(merged if merged is not None else transcripts[0].render())
