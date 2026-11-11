"""Phase 3: reconcile several imperfect transcripts into one consensus transcript with a chat model.

Global context: this reproduces, automatically, what a human editor does with multiple
speech-to-text outputs: keep wording the models agree on, prefer the timestamped/diarized
structure, mark what nobody could hear. The prompt is content-agnostic; the caller can pass
`context` (domain hint) to steer it.
Source: https://platform.openai.com/docs/api-reference/responses/create
"""
import logging

from openai import OpenAI

from .stt import Transcript

log = logging.getLogger("transcribe")

INSTRUCTIONS = """You are an expert transcript editor. You receive several automatic transcriptions of the SAME recording,
produced by different speech-to-text models. Produce one best transcript.
Rules:
- Keep the original language(s) exactly as spoken; do not translate, summarise or paraphrase.
- Where models disagree, choose the reading supported by most models or by the surrounding context.
- Preserve every utterance; do not drop content that only one model captured if it is plausible.
- Use the timestamps and speaker labels from the model that provides them; one line per utterance:
  [m:ss] SPEAKER: text
- Mark an uncertain word with [?] after it and an unintelligible passage with [~]. Never invent content.
- Output only the transcript, no commentary."""


def build_prompt(transcripts: list[Transcript], context: str | None) -> str:
    """Assemble the user message: optional domain context, then each raw transcript in its own block."""
    parts = [f"Context about the recording: {context}"] if context else []
    parts += [f"===== {t.run_id} =====\n{t.render()}" for t in transcripts]
    return "\n\n".join(parts)


def ask(client: OpenAI, label: str, model: str, instructions: str, prompt: str) -> str:
    """One Responses API call with the full request and output logged (shared by merge and correct)."""
    log.info("%s: model=%s\n--- instructions ---\n%s\n--- input ---\n%s", label, model, instructions, prompt)
    resp = client.responses.create(model=model, instructions=instructions, input=prompt)
    log.info("%s: usage=%s\n%s: output:\n%s", label, resp.usage, label, resp.output_text)
    return resp.output_text


def consensus(client: OpenAI, transcripts: list[Transcript], model: str, context: str | None) -> str:
    """Call the Responses API once and return the merged transcript text."""
    return ask(client, "merge", model, INSTRUCTIONS, build_prompt(transcripts, context))
