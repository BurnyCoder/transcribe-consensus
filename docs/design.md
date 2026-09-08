# Design notes

## Why several models instead of one

On a bilingual, two-speaker screen recording, single runs of `gpt-4o-transcribe` produced fluent
text but silently dropped sentences and misheard proper nouns; `whisper-1` kept more of the
dialogue and, when pinned to the right language, spelled names better, but had no speaker labels;
`gpt-4o-transcribe-diarize` gave speakers and timestamps but garbled overlapped speech. No single
run was best everywhere, and their errors were largely uncorrelated. Feeding all of them to a chat
model with editing rules produced a transcript closer to a human edit than any input.

## Why language pinning is a run dimension

`language` is a strong prior for Whisper-family models. For code-switched recordings, running once
per plausible language and letting the merge pick per utterance beats forcing a single language.
The diarize model does not accept `language`, so it is still run once per language only to keep run
ids unique and the loop simple; that costs an extra call and can be avoided with `--models`.

## Why the Responses API for the merge

It is OpenAI's current primary text interface and works with both non-reasoning (`gpt-4.1`) and
reasoning (`gpt-5`) models through the same call. The merge is one request with all transcripts
inline; recordings of typical meeting length fit comfortably within context.

## Why ffmpeg segmenting rather than server-side chunking

Only the diarize model exposes `chunking_strategy`, and the 25 MB upload limit applies regardless.
Stream-copy segmenting is fast, and measuring each part's real duration with ffprobe keeps
timestamps aligned when they are stitched back together.

## Trade-offs

- Sequential API calls keep the code simple; parallelising per model would cut wall time.
- Timestamps in the consensus come from the models that provide them, so the merge quality of
  timing depends on including `whisper-1` or the diarize model.
- The merge is a single LLM call. For very long recordings a windowed merge would be needed.
