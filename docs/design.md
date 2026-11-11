# Design notes

## Why several models instead of one

On a bilingual, two-speaker screen recording, single runs of `gpt-4o-transcribe` produced fluent
text but silently dropped sentences and misheard proper nouns; `whisper-1` kept more of the
dialogue and, when pinned to the right language, spelled names better, but had no speaker labels;
`gpt-4o-transcribe-diarize` gave speakers and timestamps but garbled overlapped speech. No single
run was best everywhere, and their errors were largely uncorrelated. Feeding all of them to a chat
model with editing rules produced a transcript closer to a human edit than any input.

## Why gpt-transcribe plus two deprecated models

OpenAI's speech-to-text guide recommends `gpt-transcribe` for recorded speech, and the deprecations
page schedules every older transcription model for shutdown on 2027-02-26. `gpt-transcribe` returns
text only. `whisper-1` is still the only documented source of segment timestamps, and
`gpt-4o-transcribe-diarize` the only documented source of speaker labels. The default set therefore
pairs the newest text model with the two models that carry structure. `gpt-4o-transcribe` was
dropped from the defaults because its deprecation entry names `gpt-transcribe` as its replacement.

## How language hints become runs

`language` is a strong prior for Whisper-family models. For code-switched recordings, running once
per plausible language and letting the merge pick per utterance beats forcing a single language.
`gpt-transcribe` instead takes a `languages` list and must not also receive `language`, so it gets
every hint in one run. Models that accept no language (the diarize model) run once.
`stt.language_runs` encodes this rule.

## Why a separate correction phase

The merge chooses between readings, but it keeps errors that all models share and those it cannot
resolve, such as a similar-sounding wrong word that makes no sense in context. A second pass that
reads the whole consensus for meaning can catch those. The pass gets the raw transcripts as
alternative readings because work on LLM ASR error correction uses N-best hypotheses as input
rather than a single transcript (HyPoradise, https://arxiv.org/abs/2309.15701; Ma et al.,
https://arxiv.org/abs/2307.04172). Two rules limit the risk of rewriting correct speech: lines,
timestamps and speakers must stay unchanged, and spoken style (fillers, dialect, code-switching)
must not be polished. Because the line structure is fixed, a line-level `difflib` diff of consensus
to corrected text is a complete audit trail of the changes.

## Why the Responses API for the merge

It is OpenAI's current primary text interface and works with non-reasoning and reasoning models
through the same call. The merge and correction are one request each, with all transcripts inline.
The default `gpt-6-astra` has a 1,050,000-token context window, so recordings of typical meeting
length fit comfortably.

## Why ffmpeg segmenting rather than server-side chunking

Only the diarize model exposes `chunking_strategy`, and the 25 MB upload limit applies regardless.
Stream-copy segmenting is fast, and measuring each part's real duration with ffprobe keeps
timestamps aligned when they are stitched back together.

## Trade-offs

- Sequential API calls keep the code simple; parallelising per model would cut wall time.
- Timestamps in the consensus come from the models that provide them, so the merge quality of
  timing depends on including `whisper-1` or the diarize model.
- The merge and correction are single LLM calls. For very long recordings a windowed merge would be needed.
- The correction step has no ground-truth evaluation in this repo. Its value on a given recording
  can only be judged by reviewing `corrections.diff` against the audio.
