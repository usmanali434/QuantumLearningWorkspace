"""Split long documents into ~200-300 word chunks for embedding."""

from pathlib import Path


TARGET_MIN_WORDS = 200
TARGET_MAX_WORDS = 300

# Map filenames to short source slugs used in chunk ids (pdf_chunk_0, yt_chunk_1, ...)
SOURCE_SLUGS: dict[str, str] = {
    "photosynthesis_overview.txt": "pdf",
    "youtube_lecture_energy.txt": "yt",
    "conflict_notes.txt": "conflict",
    "injection_sample.txt": "injection",
}


def load_text(path: str | Path) -> str:
    """Load a UTF-8 text file and normalize line endings."""
    return Path(path).read_text(encoding="utf-8").strip()


def source_slug_for_file(path: str | Path) -> str:
    """Return a short slug for chunk id prefixes."""
    name = Path(path).name
    if name in SOURCE_SLUGS:
        return SOURCE_SLUGS[name]
    stem = Path(path).stem.lower().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in stem)[:24] or "doc"


def _word_count(text: str) -> int:
    return len(text.split())


def _split_oversized(paragraph: str, max_words: int = TARGET_MAX_WORDS) -> list[str]:
    """Split a single oversized paragraph into word windows of max_words."""
    words = paragraph.split()
    if len(words) <= max_words:
        return [paragraph.strip()] if paragraph.strip() else []

    pieces = []
    for start in range(0, len(words), max_words):
        piece = " ".join(words[start : start + max_words]).strip()
        if piece:
            pieces.append(piece)
    return pieces


def chunk_text(
    text: str,
    source: str = "",
    min_words: int = TARGET_MIN_WORDS,
    max_words: int = TARGET_MAX_WORDS,
    id_prefix: str = "chunk",
) -> list[dict]:
    """
    Chunk text by blank-line paragraphs, merging short ones and splitting long ones
    so each chunk lands roughly between min_words and max_words.

    Chunk ids become ``{id_prefix}_{index}`` (e.g. pdf_chunk_0, yt_chunk_0).
    """
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraphs: list[str] = []
    for para in raw_paragraphs:
        paragraphs.extend(_split_oversized(para, max_words=max_words))

    merged: list[str] = []
    buffer = ""

    for para in paragraphs:
        candidate = f"{buffer}\n\n{para}".strip() if buffer else para
        if buffer and _word_count(candidate) > max_words:
            merged.append(buffer)
            buffer = para
        else:
            buffer = candidate
            if _word_count(buffer) >= min_words:
                if _word_count(buffer) >= max_words:
                    merged.append(buffer)
                    buffer = ""

    if buffer:
        if merged and _word_count(buffer) < min_words:
            combined = f"{merged[-1]}\n\n{buffer}".strip()
            if _word_count(combined) <= max_words + (min_words // 2):
                merged[-1] = combined
            else:
                merged.append(buffer)
        else:
            merged.append(buffer)

    chunks = []
    for index, chunk_body in enumerate(merged):
        slug = id_prefix.replace("_chunk", "") if id_prefix.endswith("_chunk") else id_prefix
        chunks.append(
            {
                "id": f"{id_prefix}_{index}",
                "text": chunk_body,
                "metadata": {
                    "source": source,
                    "chunk_index": index,
                    "word_count": _word_count(chunk_body),
                    "source_slug": slug,
                },
            }
        )
    return chunks


def chunk_file(
    path: str | Path,
    min_words: int = TARGET_MIN_WORDS,
    max_words: int = TARGET_MAX_WORDS,
    id_prefix: str | None = None,
) -> list[dict]:
    """Load a file and return chunk dicts with source metadata and prefixed ids."""
    file_path = Path(path)
    text = load_text(file_path)
    slug = source_slug_for_file(file_path)
    prefix = id_prefix or f"{slug}_chunk"
    return chunk_text(
        text,
        source=file_path.name,
        min_words=min_words,
        max_words=max_words,
        id_prefix=prefix,
    )


def chunk_data_directory(
    data_dir: str | Path,
    min_words: int = TARGET_MIN_WORDS,
    max_words: int = TARGET_MAX_WORDS,
    fixture_min_words: int = 40,
) -> list[dict]:
    """
    Chunk every ``.txt`` file in a directory.

    Short fixture files (conflict / injection / youtube) use a lower min_words so
    contradictory or attack passages stay as separate chunks.
    """
    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Data directory not found: {directory}")

    short_fixtures = {
        "conflict_notes.txt",
        "injection_sample.txt",
        "youtube_lecture_energy.txt",
    }
    all_chunks: list[dict] = []
    for path in sorted(directory.glob("*.txt")):
        words = fixture_min_words if path.name in short_fixtures else min_words
        all_chunks.extend(chunk_file(path, min_words=words, max_words=max_words))
    return all_chunks
