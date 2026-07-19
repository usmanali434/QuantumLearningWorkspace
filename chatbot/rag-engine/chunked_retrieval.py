"""
Part A demo: chunk a real document, retrieve top-k, relevance gate, call Groq.

Usage:
  python chunked_retrieval.py
  python chunked_retrieval.py --combine
  python chunked_retrieval.py --k 4 --question "What is chlorophyll?"
"""

from __future__ import annotations

import argparse
import sys

from rag_service import REFUSAL_MESSAGE, AskResult, create_engine, ask
from vector_store import DEFAULT_TOP_K

DEFAULT_QUESTION = "Where do the light-dependent reactions take place?"
COMBINE_QUESTION = (
    "Compare the light-dependent reactions and the Calvin cycle: "
    "where each happens and what each produces."
)


def print_result(question: str, result: AskResult) -> None:
    print(f"\nQuestion: {question}")
    print(f"top_k={result.top_k} refused={result.refused}")
    if result.sources:
        print(f"\nRetrieved {len(result.sources)} chunk(s):")
        for src in result.sources:
            dist = f", distance={src.distance:.4f}" if src.distance is not None else ""
            print(f"  - {src.id}{dist}: {src.preview}")
    print("\nLLM Final Answer:\n" if not result.refused else "\nRefusal:\n")
    print(result.answer)
    if result.refused and result.answer != REFUSAL_MESSAGE:
        print(f"(expected refusal text: {REFUSAL_MESSAGE!r})")


def run(question: str, n_results: int) -> None:
    print("Warming RAG engine (chunk -> embed -> Chroma)...")
    engine = create_engine(collection_name="study_chunks")
    print(f"Indexed {engine.chunks_indexed} chunk(s); max_distance={engine.max_distance}")
    result = ask(engine, question, top_k=n_results, include_sources=True)
    print_result(question, result)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunked RAG retrieval demo")
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Override the default test question",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Ask a multi-section compare question (pairs well with top-k >= 2)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.combine:
        question = args.question or COMBINE_QUESTION
    else:
        question = args.question or DEFAULT_QUESTION
    run(question=question, n_results=max(1, args.k))


if __name__ == "__main__":
    main(sys.argv[1:])
