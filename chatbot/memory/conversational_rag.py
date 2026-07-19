"""
Multi-turn conversational RAG with short-term memory (CLI demo).

Default mode runs a scripted 3-4 turn conversation (reviewer-friendly).
Pass --interactive for a live REPL.

Usage:
  python conversational_rag.py
  python conversational_rag.py --interactive
  python conversational_rag.py --k 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow importing helpers from ../rag-engine when run as a script
RAG_ENGINE_DIR = Path(__file__).resolve().parent.parent / "rag-engine"
if str(RAG_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_ENGINE_DIR))

from rag_service import create_engine, ask  # noqa: E402
from vector_store import DEFAULT_TOP_K  # noqa: E402

SCRIPTED_TURNS = [
    "What is photosynthesis?",
    "Where does it happen in the plant?",
    "What about the second stage you mentioned?",
    "Why does that matter for Earth's atmosphere?",
]


def run_scripted(engine, n_results: int) -> None:
    history: list[dict] = []
    print("\n=== Scripted conversation demo ===\n")
    for turn_number, question in enumerate(SCRIPTED_TURNS, start=1):
        print(f"\n----- Turn {turn_number} -----")
        print(f"User: {question}")
        result = ask(
            engine,
            question,
            history=history,
            top_k=n_results,
            include_sources=True,
            update_history=True,
        )
        if result.sources:
            print(f"Retrieved {len(result.sources)} chunk(s) (refused={result.refused}):")
            for src in result.sources:
                dist = f", d={src.distance:.4f}" if src.distance is not None else ""
                print(f"  - {src.id}{dist}")
        print(f"\nAssistant: {result.answer}")


def run_interactive(engine, n_results: int) -> None:
    history: list[dict] = []
    print("\n=== Interactive mode (type 'quit' to exit) ===\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break
        result = ask(
            engine,
            question,
            history=history,
            top_k=n_results,
            include_sources=False,
            update_history=True,
        )
        print(f"\nAssistant: {result.answer}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conversational RAG with memory")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Live REPL instead of the scripted demo",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve per turn (default: {DEFAULT_TOP_K})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print("Warming RAG engine...")
    engine = create_engine(collection_name="study_chunks_memory")
    print(f"Indexed {engine.chunks_indexed} chunk(s); max_distance={engine.max_distance}")
    n_results = max(1, args.k)

    if args.interactive:
        run_interactive(engine, n_results)
    else:
        run_scripted(engine, n_results)


if __name__ == "__main__":
    main(sys.argv[1:])
