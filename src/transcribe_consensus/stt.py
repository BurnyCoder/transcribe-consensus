"""Phase 2: run one or more OpenAI speech-to-text models over the audio chunks.

Global context: different models make different mistakes, so we collect several independent
transcripts (optionally with per-language hints) and let Phase 3 reconcile them. Each model
has a small capability record because the endpoint accepts different parameters per model.
Sources: https://platform.openai.com/docs/guides/speech-to-text
         https://platform.openai.com/docs/api-reference/audio/createTranscription
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

log = logging.getLogger("transcribe")


@dataclass(frozen=True)
class ModelSpec:
    """What one transcription model supports (drives which kwargs we send)."""

    name: str
    response_format: str  # "verbose_json" (timestamps), "diarized_json" (speakers), or "json"
    language: bool = True  # accepts ISO-639-1 `language`
    prompt: bool = True  # accepts vocabulary/style `prompt`
    extra: dict = field(default_factory=dict)  # model-specific kwargs


# Model table. whisper-1 is the only one returning segment timestamps in verbose_json;
# gpt-4o-transcribe(-mini) return plain text; the diarize model returns speaker-labelled segments.
MODELS: dict[str, ModelSpec] = {
    "whisper-1": ModelSpec("whisper-1", "verbose_json"),
    "gpt-4o-transcribe": ModelSpec("gpt-4o-transcribe", "json"),
    "gpt-4o-mini-transcribe": ModelSpec("gpt-4o-mini-transcribe", "json"),
    "gpt-4o-transcribe-diarize": ModelSpec(
        "gpt-4o-transcribe-diarize", "diarized_json", language=False, prompt=False,
        extra={"chunking_strategy": "auto"},  # server-side VAD chunking, required by the diarize model
    ),
}


@dataclass
class Segment:
    start: float
    text: str
    speaker: str = ""


@dataclass
class Transcript:
    """One model's output over the whole recording."""

    run_id: str  # e.g. "whisper-1[cs]" so the same model with two languages can coexist
    text: str
    segments: list[Segment] = field(default_factory=list)

    def render(self) -> str:
        """Human-readable form: timestamped/speaker lines when we have segments, else the plain text."""
        if not self.segments:
            return self.text
        return "\n".join(f"[{s.start:7.1f}] {s.speaker + ': ' if s.speaker else ''}{s.text.strip()}" for s in self.segments)


def _parse(resp, offset: float) -> tuple[str, list[Segment]]:
    """Normalise whatever the API returned (str, json, verbose_json, diarized_json) to (text, segments)."""
    if isinstance(resp, str):
        return resp, []
    d = resp.model_dump()
    segs = [Segment(float(s.get("start", 0)) + offset, s["text"], s.get("speaker", "")) for s in d.get("segments") or []]
    return d.get("text", ""), segs


def transcribe(client: OpenAI, spec: ModelSpec, chunks: list[tuple[Path, float]], language: str | None, prompt: str | None) -> Transcript:
    """Send every chunk to `spec.name`, concatenating texts and offsetting segment times by chunk start."""
    kwargs = {"model": spec.name, "response_format": spec.response_format, **spec.extra}
    if language and spec.language:
        kwargs["language"] = language
    if prompt and spec.prompt:
        kwargs["prompt"] = prompt
    run_id = f"{spec.name}[{language}]" if language else spec.name
    log.info("%s: request kwargs=%s", run_id, kwargs)
    texts, segments = [], []
    for path, offset in chunks:
        with path.open("rb") as f:
            resp = client.audio.transcriptions.create(file=f, **kwargs)  # the actual API call
        text, segs = _parse(resp, offset)
        texts.append(text)
        segments.extend(segs)
        log.info("%s: chunk %s (+%.1fs) -> %d chars, %d segments", run_id, path.name, offset, len(text), len(segs))
    t = Transcript(run_id, "\n".join(texts), segments)
    log.info("%s: full output:\n%s", run_id, t.render())  # never truncated, per logging policy
    return t
