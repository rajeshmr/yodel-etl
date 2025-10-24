#!/usr/bin/env python3
"""
extract_chat_qa_tsv.py

Query Ragflow chat API to retrieve all chat assistants and their sessions,
then write one TSV line per session with columns:

chat_id<TAB>session_id<TAB>user_question<TAB>assistant_response

Behavior changes from previous script:
- If the first message in a session is an assistant greeting, it will be removed.
- For each session we pick the first user message (the user's question) and the assistant reply that follows it.
  If multiple assistant messages appear consecutively right after the user question, they are concatenated.
- If there is no user message or no assistant reply, the corresponding field is left empty.

Usage:
  export RAGFLOW_TOKEN="ragflow-..."
  python3 extract_chat_qa_tsv.py --base-url https://...awsapprunner.com --out qa.tsv

Flags:
  --include-empty : include sessions even if user_question is empty (default: include)
  --page-size N    : page size for API calls (default 100)

"""

import os
import sys
import argparse
import requests
import time
from typing import Optional, List


def safe_text(s: Optional[str]) -> str:
    """Normalize message text so TSV stays valid: replace newlines/tabs with spaces and strip."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return s.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


class RagflowExtractor:
    def __init__(self, base_url: str, token: str, page_size: int = 100, sleep: float = 0.05):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.page_size = page_size
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self.sleep = sleep

    def fetch_chats(self):
        page = 1
        while True:
            params = {"page_size": self.page_size, "page": page}
            url = f"{self.base_url}/api/v1/chats"
            r = requests.get(url, headers=self.headers, params=params, timeout=30)
            r.raise_for_status()
            payload = r.json()
            records = payload.get("data") or []
            if not records:
                break
            for rec in records:
                yield rec
            page += 1
            time.sleep(self.sleep)

    def fetch_sessions_for_chat(self, chat_id: str):
        page = 1
        while True:
            params = {"page_size": self.page_size, "page": page}
            url = f"{self.base_url}/api/v1/chats/{chat_id}/sessions"
            r = requests.get(url, headers=self.headers, params=params, timeout=30)
            r.raise_for_status()
            payload = r.json()
            records = payload.get("data") or []
            if not records:
                break
            for rec in records:
                yield rec
            page += 1
            time.sleep(self.sleep)


def extract_first_qa_from_session(session: dict) -> (str, str):
    """Given a session dict, remove leading assistant greeting (if present),
    then return tuple (user_question, assistant_response).

    Strategy:
    - If the first message has role 'assistant', drop it (assumed greeting).
    - Find the first message with role 'user' -> user_question.
    - Collect the following assistant messages (one or more contiguous) as assistant_response.
    - Normalize and return both strings.
    """
    messages = session.get("messages") or []
    if isinstance(messages, dict):
        messages = list(messages.values())

    # Normalize message roles and contents
    msgs = []
    for m in messages:
        role = m.get("role") or m.get("type") or ""
        content = m.get("content")
        msgs.append({"role": role, "content": content})

    # Remove first assistant greeting if it's the very first message
    if msgs and msgs[0]["role"] == "assistant":
        msgs = msgs[1:]

    # Find first user message
    user_idx = None
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            user_idx = i
            break

    if user_idx is None:
        return ("", "")

    user_q = safe_text(msgs[user_idx]["content"])

    # Find assistant messages immediately after user_idx
    assistant_parts: List[str] = []
    j = user_idx + 1
    while j < len(msgs) and msgs[j]["role"] == "assistant":
        assistant_parts.append(safe_text(msgs[j]["content"]))
        j += 1

    assistant_resp = " ".join([p for p in assistant_parts if p])
    return (user_q, assistant_resp)


def main():
    ap = argparse.ArgumentParser(description="Extract first user Q and assistant response per session to TSV")
    ap.add_argument("--base-url", required=True, help="Base URL e.g. https://...awsapprunner.com")
    ap.add_argument("--token", default=os.environ.get("RAGFLOW_TOKEN"), help="Bearer token (or set RAGFLOW_TOKEN)")
    ap.add_argument("--page-size", type=int, default=100, help="Page size for API calls")
    ap.add_argument("--out", default="qa.tsv", help="Output TSV file")
    ap.add_argument("--include-empty", action="store_true", help="Include sessions with empty user_question or assistant_response")
    args = ap.parse_args()

    if not args.token:
        print("ERROR: No token provided. Set RAGFLOW_TOKEN or use --token.", file=sys.stderr)
        sys.exit(2)

    extractor = RagflowExtractor(args.base_url, args.token, page_size=args.page_size)

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("chat_id	session_id	user_question	assistant_response\n")
        total_sessions = 0
        total_written = 0
        for chat in extractor.fetch_chats():
            chat_id = chat.get("id") or ""
            for session in extractor.fetch_sessions_for_chat(chat_id):
                total_sessions += 1
                session_id = session.get("id") or ""
                user_q, assistant_resp = extract_first_qa_from_session(session)
                if not args.include_empty and not user_q and not assistant_resp:
                    continue
                line = f"{chat_id}	{session_id}	{user_q}	{assistant_resp}\n"
                fh.write(line)
                total_written += 1

    print(f"Done. Sessions scanned: {total_sessions}, lines written: {total_written}")
    print(f"Output file: {args.out}")


if __name__ == "__main__":
    main()
