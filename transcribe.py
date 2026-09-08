"""Thin launcher so `uv run transcribe.py <file>` works without installing the package."""
from transcribe_consensus import main

if __name__ == "__main__":
    main()
