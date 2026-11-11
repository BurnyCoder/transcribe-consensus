"""Phase 4: proofread the consensus transcript for words that were most likely misheard.

Global context: the merge (phase 3) picks between readings but still keeps errors that every model
shared, or that it could not resolve, e.g. a similar-sounding wrong word that makes no sense in
context. This phase asks a chat model to fix only those errors. The raw model outputs are passed
as alternative readings because LLM ASR error correction works better from several hypotheses
than from one (HyPoradise, https://arxiv.org/abs/2309.15701; Ma et al., https://arxiv.org/abs/2307.04172).
The rules keep line structure fixed, so `corrections.diff` (written by the orchestrator) shows
every change for review.
"""
from openai import OpenAI

from .merge import ask, build_prompt
from .stt import Transcript

# Content-agnostic proofreading rules (domain hints go through --context, never into this text).
INSTRUCTIONS = """You are a careful proofreader of automatic speech-recognition (ASR) transcripts. You receive a
TRANSCRIPT TO CORRECT (an edited consensus of several ASR models), followed by the raw outputs of those
models as alternative readings of the same audio.
Find and fix words that were most likely misheard. Typical ASR errors: a word replaced by a similar-sounding
word, wrong word boundaries, misspelled names or technical terms, a word recognised in the wrong language in
code-switched speech, a wrong word ending that breaks the sentence, a phrase that makes no sense in context.
Rules:
- Change words only where the text as written does not make sense in context and a similar-sounding
  alternative does. Prefer alternatives that appear in one of the raw readings.
- Do not polish speech: keep filler words, false starts, repetitions, colloquial and dialect forms, slang and
  ungrammatical but plausible spoken phrasing. Keep each utterance in the language or mix of languages it was
  spoken in; never translate or normalise one language into another.
- Keep every line, timestamp and speaker label exactly as given; do not merge, split, reorder, add or drop lines.
- Append [?] after a word you changed when you are not confident. Keep existing [?] and [~] markers unless the
  raw readings clearly resolve them. Never invent content.
- Output only the corrected transcript, no commentary."""


def build_correction_prompt(text: str, transcripts: list[Transcript], context: str | None) -> str:
    """User message: optional context, the transcript to correct, then each raw reading as evidence."""
    head = [f"Context about the recording: {context}"] if context else []
    return "\n\n".join(head + [f"===== TRANSCRIPT TO CORRECT =====\n{text}", build_prompt(transcripts, None)])


def correct(client: OpenAI, text: str, transcripts: list[Transcript], model: str, context: str | None) -> str:
    """Call the Responses API once and return the corrected transcript."""
    return ask(client, "correct", model, INSTRUCTIONS, build_correction_prompt(text, transcripts, context))
