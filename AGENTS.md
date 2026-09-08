# AGENTS.md

Guidance for people and coding agents working on this repo. Usage and methodology live in
`README.md`; do not duplicate them here.

## Layout

```
transcribe.py                      launcher: `uv run transcribe.py <file>`
src/transcribe_consensus/
  __init__.py                      CLI parsing, phase orchestration, output writing, raw.json reload
  audio.py                         phase 1: ffmpeg extraction, size check, segment splitting, offsets
  stt.py                           phase 2: model capability table, API calls, Transcript/Segment types
  merge.py                         phase 3: consensus instructions, prompt builder, Responses API call
  log.py                           shared timestamped logger (terminal + logs/<timestamp>.log)
tests/test_pipeline.py             offline tests (synthetic ffmpeg tone, fake OpenAI client)
docs/design.md                     design decisions and trade-offs
.env.example                       every TRANSCRIBE_* setting with a comment
```

## Conventions

- Keep phases pure functions that take a client/paths and return data; only `main()` touches argv, `.env` and stdout.
- Adding a model means one `ModelSpec` row in `stt.py`. Set `language`/`prompt` to `False` if the API rejects those parameters for it and put required extra kwargs in `extra`.
- The merge instructions in `merge.py` must stay content-agnostic. Domain hints belong in `--prompt` (speech-to-text) and `--context` (merge), never in code.
- Everything sent to or received from the API is logged in full through the `transcribe` logger. Do not truncate.
- `output/`, `logs/`, `.env` and media files are git-ignored. Never commit a key.

## Working on it

```bash
uv sync
uv run pytest                      # must stay green and offline
uv run transcribe.py sample.mp4    # end-to-end check needs OPENAI_API_KEY in .env
```

Use a feature branch and a pull request per change.
