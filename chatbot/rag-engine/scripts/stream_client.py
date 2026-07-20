#!/usr/bin/env python3
"""
Test client for POST /ask/stream (NDJSON).

Usage:
  python scripts/stream_client.py "Where does the Calvin cycle occur?"
  python scripts/stream_client.py --url http://127.0.0.1:8001/ask/stream "What is photosynthesis?"
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def stream_ask(url: str, question: str) -> int:
    payload = json.dumps({"question": question, "include_sources": True}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print(f"POST {url}")
    print(f"Question: {question}\n")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                event = json.loads(line)
                etype = event.get("type")
                if etype == "metadata":
                    print(
                        f"[metadata] refused={event.get('refused')} "
                        f"rounds={event.get('retrieval_rounds')} "
                        f"cached={event.get('cached')} "
                        f"sources={event.get('source_ids')}"
                    )
                    if event.get("refused") and event.get("answer"):
                        print(f"\nRefusal: {event['answer']}\n")
                elif etype == "token":
                    print(event.get("content", ""), end="", flush=True)
                elif etype == "done":
                    print("\n")
                    print(
                        f"[done] grounded={event.get('grounded')} "
                        f"cached={event.get('cached')} "
                        f"timing={event.get('timing')}"
                    )
                elif etype == "error":
                    print(f"\n[error] {event.get('detail')}", file=sys.stderr)
                    return 1
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Connection error: {exc}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream client for /ask/stream")
    parser.add_argument("question", nargs="?", default="Where does the Calvin cycle occur?")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8001/ask/stream",
        help="Full URL to the stream endpoint",
    )
    args = parser.parse_args()
    return stream_ask(args.url, args.question)


if __name__ == "__main__":
    raise SystemExit(main())
