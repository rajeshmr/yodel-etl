#!/usr/bin/env python3
"""
Test chat assistant by asking questions from a file and recording responses.
Each question creates a new session to avoid chat history affecting results.

Added: after receiving assistant answer + references, call Claude (Anthropic)
as a judge to classify the (question, answer, references) triple. The judge's
JSON output is stored in the CSV along with raw text fallback.
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration for Chat API (unchanged)
API_BASE_URL = os.environ.get("RAGFLOW_BASE_URL", "http://localhost:9380/api/v1/chats")
CHAT_ID = os.environ.get("RAGFLOW_CHAT_ID", "xxxx")
API_TOKEN = os.environ.get("RAGFLOW_TOKEN", "ragflow-xxxx")

# Configuration for Anthropic / Claude judge
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL_DEFAULT = "claude-sonnet-4-5"  # override with --anthropic-model arg if desired
ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"


def parse_sse_stream(response) -> Tuple[Optional[str], Optional[Dict], Optional[str], int]:
    """
    Parse Server-Sent Events stream and extract final answer and references.

    Returns:
        Tuple of (answer, references_dict, session_id, status_code)
    """
    answer = None
    references = None
    session_id = None
    status_code = response.status_code

    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            # Many SSE servers include "data:" lines; only parse those.
            if not line.startswith("data:"):
                continue

            # Remove "data:" prefix
            json_str = line[5:].strip()

            # Skip sentinel tokens like [DONE] or empty data
            if not json_str or json_str in ("[DONE]", "null", "None"):
                continue

            try:
                data = json.loads(json_str)

                # Check for error code
                if isinstance(data, dict) and data.get("code") != 0:
                    print(
                        f"API error code: {data.get('code')}, message: {data.get('message')}",
                        file=sys.stderr,
                    )
                    continue

                # Extract data payload
                payload = data.get("data") if isinstance(data, dict) else None

                # Skip boolean completion markers
                if isinstance(payload, bool):
                    continue

                if isinstance(payload, dict):
                    # Update answer (streaming chunks). We overwrite so final chunk persists.
                    if "answer" in payload:
                        # If API streams incremental deltas, consider concatenating instead:
                        # answer = (answer or "") + payload["answer"]
                        answer = payload["answer"]

                    # Update session_id
                    if "session_id" in payload:
                        session_id = payload["session_id"]

                    # Update references (usually comes with final answer)
                    if "reference" in payload:
                        references = payload["reference"]

            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {json_str[:200]}... Error: {e}", file=sys.stderr)
                continue

    except Exception as e:
        print(f"Error parsing stream: {e}", file=sys.stderr)

    return answer, references, session_id, status_code


def create_session() -> Tuple[Optional[str], int]:
    """
    Create a new chat session by sending an empty question.
    Returns: (session_id, status_code)
    """
    url = f"{API_BASE_URL}/{CHAT_ID}/completions"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"question": "", "stream": True}

    try:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
        # handle non-2xx early
        if response.status_code >= 400:
            print(f"Failed to create session HTTP {response.status_code}: {response.text[:200]}", file=sys.stderr)
            return None, response.status_code

        answer, references, session_id, status_code = parse_sse_stream(response)
        return session_id, status_code

    except requests.exceptions.RequestException as e:
        print(f"Failed to create session: {e}", file=sys.stderr)
        return None, 0


def ask_question(question: str, session_id: str) -> Tuple[Optional[str], Optional[Dict], int]:
    """
    Ask a question in an existing session.
    Returns: (answer, references_dict, status_code)
    """
    url = f"{API_BASE_URL}/{CHAT_ID}/completions"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"question": question, "stream": True, "session_id": session_id}

    try:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
        if response.status_code >= 400:
            print(f"Ask question HTTP {response.status_code}: {response.text[:200]}", file=sys.stderr)
            return None, None, response.status_code

        answer, references, _, status_code = parse_sse_stream(response)
        return answer, references, status_code

    except requests.exceptions.RequestException as e:
        print(f"Failed to ask question: {e}", file=sys.stderr)
        return None, None, 0


def extract_reference_info(references: Optional[Dict]) -> Dict[str, any]:
    """
    Extract reference chunk information into separate fields.
    """
    if not references or "chunks" not in references:
        return {
            "total_references": 0,
            "chunk_ids": "",
            "document_names": "",
            "similarities": "",
            "vector_similarities": "",
            "term_similarities": "",
        }

    chunks = references.get("chunks", [])
    total = references.get("total", len(chunks))

    chunk_ids = []
    document_names = []
    similarities = []
    vector_sims = []
    term_sims = []

    for chunk in chunks:
        chunk_ids.append(str(chunk.get("id") or ""))
        document_names.append(str(chunk.get("document_name") or ""))
        similarities.append(str(chunk.get("similarity") or ""))
        vector_sims.append(str(chunk.get("vector_similarity") or ""))
        term_sims.append(str(chunk.get("term_similarity") or ""))

    return {
        "total_references": total,
        "chunk_ids": ", ".join(chunk_ids),
        "document_names": ", ".join(document_names),
        "similarities": ", ".join(similarities),
        "vector_similarities": ", ".join(vector_sims),
        "term_similarities": ", ".join(term_sims),
    }


def build_judge_prompt(question: str, answer: str, references: Optional[Dict], session_id: str) -> str:
    """
    Build the deterministic prompt to send to the judge LLM.
    We instruct the model to return a compact JSON. Keep it short.
    """
    # Include full chunk content for comprehensive evaluation
    chunks = []
    if references and isinstance(references, dict):
        for c in references.get("chunks", [])[:12]:  # limit to first 12 chunks to avoid huge prompts
            content = c.get("content", "")
            chunks.append(
                {
                    "id": c.get("id"),
                    "document_name": c.get("document_name"),
                    "similarity": c.get("similarity"),
                    "content": content,
                }
            )

    prompt = f"""
You are an evaluator that must judge whether an assistant answer correctly responds to a user's question,
and whether the retrieved reference chunks support the answer.

Return ONLY a single valid JSON object (no surrounding text) with fields:
- labels: dict of boolean flags (answer_present, answer_direct, answer_partial, answer_wrong,
  answer_hallucination, answer_contradicts_sources, retrieval_relevant, retrieval_supports_answer,
  retrieval_complete, source_divergence, citation_ok, answer_safe, answer_requires_followup)
- scores: dict with numeric scores between 0 and 1 for answer_quality, retrieval_relevance, evidence_support
- notes: short string explanation if any problem detected (<= 200 chars)

Rules:
- answer_present: true when assistant produced a non-empty answer.
- retrieval_relevant: true if the majority of provided chunks are topically relevant to the question.
- retrieval_supports_answer: true if chunks contain explicit facts that substantiate key claims in the assistant's answer.
- answer_hallucination: true if the assistant asserts facts not present in any provided chunk.
- answer_contradicts_sources: true if assistant's claims directly contradict one or more chunks.
- If unsure, prefer conservative (i.e., set problematic flags true).

Now evaluate the following input. Keep the JSON compact.

QUESTION:
{question}

ASSISTANT_ANSWER:
{answer}

REFERENCES (first {len(chunks)} chunks shown):
{json.dumps(chunks, ensure_ascii=False)}

SESSION_ID:
{session_id}
""".strip()

    return prompt


def call_anthropic_judge(prompt: str, model: str = ANTHROPIC_MODEL_DEFAULT, api_key: Optional[str] = None) -> Tuple[int, str]:
    """
    Call Anthropic / Claude Messages API with a deterministic prompt.
    Returns: (http_status_code, raw_text_response)
    Note: expects ANTHROPIC_API_KEY in env or via api_key arg.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ANTHROPIC_API_KEY
    if not key:
        raise RuntimeError("Anthropic API key missing. Set ANTHROPIC_API_KEY environment variable or pass via args.")

    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    body = {
        "model": model,
        "max_tokens": 800,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        resp = requests.post(ANTHROPIC_ENDPOINT, headers=headers, json=body, timeout=60)
        if resp.status_code != 200:
            return resp.status_code, resp.text
        return resp.status_code, resp.text
    except requests.RequestException as e:
        return 0, f"request-exception: {e}"


def parse_judge_output(raw_text: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Try to extract JSON from the judge's raw text. Returns (parsed_json, error_message)
    """
    if not raw_text:
        return None, "empty judge response"

    # The Anthropic response body may be a JSON object with a 'completion' field,
    # or it may be the raw completion text depending on their API wrapper.
    # Try a few reasonable parsing attempts.

    # 1) If response is JSON with keys (e.g., anthropic returns a JSON wrapper)
    try:
        top = json.loads(raw_text)
        # If wrapper contains a 'completion' or 'output', try to extract it
        if isinstance(top, dict):
            # Messages API: {'id':..., 'model':..., 'content': [{'type': 'text', 'text': '...'}]}
            if "content" in top and isinstance(top["content"], list) and top["content"]:
                # Extract text from first content block
                first_content = top["content"][0]
                if isinstance(first_content, dict) and "text" in first_content:
                    candidate = first_content["text"]
                else:
                    candidate = None
            elif "completion" in top:
                candidate = top["completion"]
            elif "output" in top:
                candidate = top["output"]
            elif "choices" in top and isinstance(top["choices"], list) and top["choices"]:
                candidate = top["choices"][0].get("text") or top["choices"][0].get("message") or top["choices"][0]
            else:
                # Maybe it's already our JSON; validate labels/scores keys
                # If this top dict looks like the judge JSON, return it.
                # Quick heuristics: presence of 'labels' or 'scores' keys
                if "labels" in top or "scores" in top:
                    return top, None
                candidate = None

            if candidate:
                # candidate may itself be a JSON string
                try:
                    parsed = json.loads(candidate)
                    return parsed, None
                except Exception:
                    # fall through to raw-text parse
                    raw_text = candidate
            else:
                # fall back to raw_text
                raw_text = json.dumps(top)
    except Exception:
        # not JSON; raw_text remains
        pass

    # 2) If the raw_text contains JSON somewhere, extract first {...} block
    start = raw_text.find("{")
    if start >= 0:
        end = raw_text.rfind("}")
        if end > start:
            snippet = raw_text[start:end + 1]
            try:
                parsed = json.loads(snippet)
                return parsed, None
            except Exception as e:
                return None, f"failed to parse inner JSON: {e}; snippet len {len(snippet)}"

    # 3) can't parse
    return None, "unable to parse judge JSON"


def judge_triple_with_claude(question: str, answer: str, references: Optional[Dict], session_id: str,
                             anthropic_model: str, judge_delay: float = 0.0) -> Tuple[int, Optional[Dict], str, Optional[str]]:
    """
    Perform judge call and return (http_status, parsed_json_or_none, raw_text, parse_error_or_none)
    """
    if judge_delay > 0:
        time.sleep(judge_delay)
    
    prompt = build_judge_prompt(question, answer or "", references or {}, session_id)
    status, raw = call_anthropic_judge(prompt, model=anthropic_model)
    parsed, parse_err = None, None
    if status == 200:
        parsed, parse_err = parse_judge_output(raw)
    else:
        parse_err = f"judge-http-{status}"

    return status, parsed, raw, parse_err


def calculate_quality_color(labels: Dict[str, bool]) -> str:
    """
    Calculate quality color based on judge labels.
    GREEN = best case (direct answer, supported by sources, no issues)
    YELLOW = medium case (partial answer or minor issues)
    RED = worst case (wrong, hallucination, contradicts sources)
    """
    if not labels:
        return "GRAY"  # No labels available
    
    # RED conditions (worst case - any critical issue)
    if labels.get("answer_wrong", False):
        return "RED"
    if labels.get("answer_hallucination", False):
        return "RED"
    if labels.get("answer_contradicts_sources", False):
        return "RED"
    if not labels.get("answer_present", True):  # No answer at all
        return "RED"
    if not labels.get("retrieval_supports_answer", True):  # Answer not supported
        return "RED"
    
    # YELLOW conditions (medium case - partial or incomplete)
    if labels.get("answer_partial", False):
        return "YELLOW"
    if not labels.get("retrieval_complete", True):  # Incomplete evidence
        return "YELLOW"
    if not labels.get("retrieval_relevant", True):  # Irrelevant chunks
        return "YELLOW"
    if labels.get("source_divergence", False):  # Conflicting sources
        return "YELLOW"
    if labels.get("answer_requires_followup", False):  # Needs clarification
        return "YELLOW"
    
    # GREEN conditions (best case - direct answer, well-supported)
    if labels.get("answer_direct", False) and labels.get("retrieval_supports_answer", False):
        return "GREEN"
    
    # Default to YELLOW if answer exists but doesn't meet green criteria
    if labels.get("answer_present", False):
        return "YELLOW"
    
    return "GRAY"


def process_questions(questions_file: str, output_csv: str, delay: float = 1.0, judge_delay: float = 0.5, anthropic_model: str = ANTHROPIC_MODEL_DEFAULT):
    """
    Process all questions from file and write results to CSV including judge results from Claude.
    """
    # Read questions
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(questions)} questions from {questions_file}")

    # Open CSV for writing
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "question",
            "answer",
            "status_code",
            "session_id",
            "total_references",
            "chunk_ids",
            "document_names",
            "similarities",
            "vector_similarities",
            "term_similarities",
            # Judge fields
            "quality_color",
            "judge_status",
            "judge_raw",
            "judge_json",
            "judge_labels",
            "judge_scores",
            "judge_notes",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process each question
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] Processing: {question[:120]}...")

            # Create new session
            session_id, session_status = create_session()
            if not session_id:
                print(f"  ❌ Failed to create session (status: {session_status})")
                writer.writerow({
                    "question": question,
                    "answer": "",
                    "status_code": session_status,
                    "session_id": "",
                    "total_references": 0,
                    "chunk_ids": "",
                    "document_names": "",
                    "similarities": "",
                    "vector_similarities": "",
                    "term_similarities": "",
                    "quality_color": "GRAY",
                    "judge_status": "",
                    "judge_raw": "",
                    "judge_json": "",
                    "judge_labels": "",
                    "judge_scores": "",
                    "judge_notes": "",
                })
                continue

            print(f"  ✓ Session created: {session_id}")

            # Ask question
            time.sleep(delay)  # Rate limiting
            answer, references, status_code = ask_question(question, session_id)

            # Extract reference info
            ref_info = extract_reference_info(references)

            # Call judge (Claude)
            try:
                judge_status, judge_parsed, judge_raw, judge_parse_err = judge_triple_with_claude(
                    question, answer or "", references or {}, session_id, anthropic_model, judge_delay
                )
            except Exception as e:
                judge_status = 0
                judge_parsed = None
                judge_raw = f"judge-call-exception: {e}"
                judge_parse_err = str(e)

            # Prepare judge fields for CSV
            judge_json_str = ""
            judge_labels_str = ""
            judge_scores_str = ""
            judge_notes = ""
            quality_color = "GRAY"

            if judge_parsed:
                try:
                    judge_json_str = json.dumps(judge_parsed, ensure_ascii=False)
                except Exception:
                    judge_json_str = str(judge_parsed)

                # Extract labels & scores if present
                if isinstance(judge_parsed, dict):
                    labels = judge_parsed.get("labels", {})
                    scores = judge_parsed.get("scores", {})
                    judge_labels_str = json.dumps(labels, ensure_ascii=False)
                    judge_scores_str = json.dumps(scores, ensure_ascii=False)
                    judge_notes = judge_parsed.get("notes", "") or ""
                    quality_color = calculate_quality_color(labels)
            else:
                # If parsing failed, include the raw output in judge_raw and note
                judge_notes = judge_parse_err or ""

            # Write to CSV
            writer.writerow({
                "question": question,
                "answer": answer or "",
                "status_code": status_code,
                "session_id": session_id,
                **ref_info,
                "quality_color": quality_color,
                "judge_status": judge_status,
                "judge_raw": (judge_raw or "")[:10000],  # cap raw text
                "judge_json": judge_json_str,
                "judge_labels": judge_labels_str,
                "judge_scores": judge_scores_str,
                "judge_notes": judge_notes,
            })

            print(f"  ✓ Answer received (status: {status_code}, refs: {ref_info['total_references']})")
            print(f"  ✓ Judge status: {judge_status}; parsed: {'yes' if judge_parsed else 'no'}; note: {judge_notes}")

            # Flush to ensure data is written
            csvfile.flush()

    print(f"\n✅ Results written to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Test chat assistant with questions and judge with Claude")
    parser.add_argument("questions_file", help="Path to text file containing questions (one per line)")
    parser.add_argument("-o", "--output", default="chat_test_results_with_judge.csv",
                        help="Output CSV file (default: chat_test_results_with_judge.csv)")
    parser.add_argument("-d", "--delay", type=float, default=1.0, help="Delay in seconds between RAGFlow requests (default: 1.0)")
    parser.add_argument("--judge-delay", type=float, default=0.5, help="Delay in seconds between Claude judge API calls (default: 0.5)")
    parser.add_argument("--anthropic-model", default=os.environ.get("ANTHROPIC_MODEL", ANTHROPIC_MODEL_DEFAULT),
                        help="Anthropic model name (default claude-sonnet-4-5)")
    args = parser.parse_args()

    try:
        process_questions(args.questions_file, args.output, args.delay, args.judge_delay, args.anthropic_model)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
