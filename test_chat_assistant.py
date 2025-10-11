#!/usr/bin/env python3
"""
Test chat assistant by asking questions from a file and recording responses.
Each question creates a new session to avoid chat history affecting results.
"""

import argparse
import csv
import json
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests


# Configuration
API_BASE_URL = "https://mijm9dav2z.us-east-1.awsapprunner.com/api/v1/chats"
CHAT_ID = "f5cd5e28a68f11f0991cd75279869ede"
API_TOKEN = "ragflow-AwYmY0ZGY3YTY5MDExZjA4ZTAxZDc1Mj"


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
            if not line or not line.startswith("data:"):
                continue
            
            # Remove "data:" prefix
            json_str = line[5:].strip()
            
            try:
                data = json.loads(json_str)
                
                # Check for error code
                if data.get("code") != 0:
                    print(f"API error code: {data.get('code')}, message: {data.get('message')}", file=sys.stderr)
                    continue
                
                # Extract data payload
                payload = data.get("data")
                
                # Skip boolean completion markers
                if isinstance(payload, bool):
                    continue
                
                if isinstance(payload, dict):
                    # Update answer (streaming chunks)
                    if "answer" in payload:
                        answer = payload["answer"]
                    
                    # Update session_id
                    if "session_id" in payload:
                        session_id = payload["session_id"]
                    
                    # Update references (usually comes with final answer)
                    if "reference" in payload:
                        references = payload["reference"]
            
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {json_str[:100]}... Error: {e}", file=sys.stderr)
                continue
    
    except Exception as e:
        print(f"Error parsing stream: {e}", file=sys.stderr)
    
    return answer, references, session_id, status_code


def create_session() -> Tuple[Optional[str], int]:
    """
    Create a new chat session by sending an empty question.
    
    Returns:
        Tuple of (session_id, status_code)
    """
    url = f"{API_BASE_URL}/{CHAT_ID}/completions"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "question": "",
        "stream": True
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
        answer, references, session_id, status_code = parse_sse_stream(response)
        return session_id, status_code
    
    except requests.exceptions.RequestException as e:
        print(f"Failed to create session: {e}", file=sys.stderr)
        return None, 0


def ask_question(question: str, session_id: str) -> Tuple[Optional[str], Optional[Dict], int]:
    """
    Ask a question in an existing session.
    
    Returns:
        Tuple of (answer, references_dict, status_code)
    """
    url = f"{API_BASE_URL}/{CHAT_ID}/completions"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "question": question,
        "stream": True,
        "session_id": session_id
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
        answer, references, _, status_code = parse_sse_stream(response)
        return answer, references, status_code
    
    except requests.exceptions.RequestException as e:
        print(f"Failed to ask question: {e}", file=sys.stderr)
        return None, None, 0


def extract_reference_info(references: Optional[Dict]) -> Dict[str, any]:
    """
    Extract reference chunk information into separate fields.
    
    Returns:
        Dict with chunk_ids, document_names, similarities, etc.
    """
    if not references or "chunks" not in references:
        return {
            "total_references": 0,
            "chunk_ids": "",
            "document_names": "",
            "similarities": "",
            "vector_similarities": "",
            "term_similarities": ""
        }
    
    chunks = references.get("chunks", [])
    total = references.get("total", len(chunks))
    
    chunk_ids = []
    document_names = []
    similarities = []
    vector_sims = []
    term_sims = []
    
    for chunk in chunks:
        # Handle None values by converting to empty string
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
        "term_similarities": ", ".join(term_sims)
    }


def process_questions(questions_file: str, output_csv: str, delay: float = 1.0):
    """
    Process all questions from file and write results to CSV.
    
    Args:
        questions_file: Path to text file with questions (one per line)
        output_csv: Path to output CSV file
        delay: Delay in seconds between requests (to avoid rate limiting)
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
            "term_similarities"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each question
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] Processing: {question[:60]}...")
            
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
                    "term_similarities": ""
                })
                continue
            
            print(f"  ✓ Session created: {session_id}")
            
            # Ask question
            time.sleep(delay)  # Rate limiting
            answer, references, status_code = ask_question(question, session_id)
            
            # Extract reference info
            ref_info = extract_reference_info(references)
            
            # Write to CSV
            writer.writerow({
                "question": question,
                "answer": answer or "",
                "status_code": status_code,
                "session_id": session_id,
                **ref_info
            })
            
            print(f"  ✓ Answer received (status: {status_code}, refs: {ref_info['total_references']})")
            
            # Flush to ensure data is written
            csvfile.flush()
    
    print(f"\n✅ Results written to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Test chat assistant with questions from a file"
    )
    parser.add_argument(
        "questions_file",
        help="Path to text file containing questions (one per line)"
    )
    parser.add_argument(
        "-o", "--output",
        default="chat_test_results.csv",
        help="Output CSV file (default: chat_test_results.csv)"
    )
    parser.add_argument(
        "-d", "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between requests (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    try:
        process_questions(args.questions_file, args.output, args.delay)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
