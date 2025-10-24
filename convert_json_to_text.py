#!/usr/bin/env python3
"""
Convert JSON files to natural language summaries using Google Gemini API.

This script reads JSON files from an input directory, sends them to Gemini API
for conversion to human-readable text, and saves the output as text files.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import google.generativeai as genai


# Configuration
DEFAULT_MODEL = "models/gemini-2.5-pro"
API_DELAY_SECONDS = 0.5
LOG_FILE = "conversion.log"

# Prompt template for Gemini
PROMPT_TEMPLATE = """You are a helpful assistant that converts structured park information in JSON format into a smooth, natural English description.

Write a detailed, factual summary that reads like a short Wikipedia article or tourism guide entry. 
The goal is to make the text sound natural, coherent, and informative for a general audience.

Follow these guidelines:
- Use full sentences and flowing paragraphs.
- Include key information such as:
  - Park name and location
  - Overview or description
  - How to get there (if available)
  - Activities visitors can do
  - Available facilities
  - Accessibility features
  - Restrictions and rules
  - Nearby or related parks
  - Any special notes like free parking, scenic views, or opening hours.
- Avoid technical terms, JSON keys, or symbols.
- Do not list field names — integrate all data into natural sentences.
- Write in a friendly but factual tone.
- Length: 150–250 words if enough data is available.

Convert the following park information into a plain English description as described above.

{json_content}
"""


def setup_logging(log_file: str) -> None:
    """Configure logging to both console and file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def initialize_gemini(api_key: Optional[str] = None) -> genai.GenerativeModel:
    """
    Initialize the Gemini API client.
    
    Args:
        api_key: Optional API key. If not provided, uses GEMINI_API_KEY env variable.
    
    Returns:
        Configured GenerativeModel instance.
    
    Raises:
        ValueError: If API key is not provided or found in environment.
    """
    if api_key:
        genai.configure(api_key=api_key)
    else:
        # Try to get from environment
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError(
                "API key not provided. Set GEMINI_API_KEY or GOOGLE_API_KEY "
                "environment variable, or pass --api-key argument."
            )
        genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(DEFAULT_MODEL)
    logging.info(f"Initialized Gemini model: {DEFAULT_MODEL}")
    return model


def load_json_file(file_path: Path) -> Optional[dict]:
    """
    Load and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file.
    
    Returns:
        Parsed JSON data as dict, or None if loading fails.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded: {file_path.name}")
        return data
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {file_path.name}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error reading {file_path.name}: {e}")
        return None


def convert_json_to_text(model: genai.GenerativeModel, json_data: dict) -> Optional[str]:
    """
    Send JSON data to Gemini API and get natural language summary.
    
    Args:
        model: Configured Gemini model instance.
        json_data: JSON data to convert.
    
    Returns:
        Natural language summary, or None if conversion fails.
    """
    try:
        # Format the JSON nicely for the prompt
        json_content = json.dumps(json_data, indent=2)
        prompt = PROMPT_TEMPLATE.format(json_content=json_content)
        
        # Generate content
        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip()
        else:
            logging.error("Empty response from Gemini API")
            return None
            
    except Exception as e:
        logging.error(f"Error calling Gemini API: {e}")
        return None


def save_text_file(output_path: Path, content: str) -> bool:
    """
    Save text content to a file.
    
    Args:
        output_path: Path where to save the file.
        content: Text content to save.
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logging.info(f"Saved output: {output_path.name}")
        return True
    except Exception as e:
        logging.error(f"Error writing {output_path.name}: {e}")
        return False


def process_directory(input_dir: Path, output_dir: Path, model: genai.GenerativeModel) -> tuple[int, int, int]:
    """
    Process all JSON files in the input directory.
    
    Args:
        input_dir: Directory containing input JSON files.
        output_dir: Directory where output text files will be saved.
        model: Configured Gemini model instance.
    
    Returns:
        Tuple of (total_files, successful, failed) counts.
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory ready: {output_dir}")
    
    # Find all JSON files
    json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        logging.warning(f"No JSON files found in {input_dir}")
        return 0, 0, 0
    
    logging.info(f"Found {len(json_files)} JSON files to process")
    
    successful = 0
    failed = 0
    
    for json_file in json_files:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing: {json_file.name}")
        logging.info(f"{'='*60}")
        
        # Load JSON
        json_data = load_json_file(json_file)
        if json_data is None:
            failed += 1
            continue
        
        # Convert to text using Gemini
        text_summary = convert_json_to_text(model, json_data)
        if text_summary is None:
            failed += 1
            continue
        
        # Save output
        output_file = output_dir / f"{json_file.stem}.txt"
        if save_text_file(output_file, text_summary):
            successful += 1
        else:
            failed += 1
        
        # Rate limiting delay
        if json_file != json_files[-1]:  # Don't delay after the last file
            time.sleep(API_DELAY_SECONDS)
    
    return len(json_files), successful, failed


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert JSON files to natural language summaries using Gemini API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input-dir ./input_json --output-dir ./output_text
  %(prog)s -i data/parks -o summaries --api-key YOUR_API_KEY
  
Environment Variables:
  GEMINI_API_KEY or GOOGLE_API_KEY - API key for Google Gemini
        """
    )
    
    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        default='./input_json',
        help='Input directory containing JSON files (default: ./input_json)'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='./output_text',
        help='Output directory for text files (default: ./output_text)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='Gemini API key (or set GEMINI_API_KEY environment variable)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default=LOG_FILE,
        help=f'Log file path (default: {LOG_FILE})'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    
    logging.info("="*60)
    logging.info("JSON to Text Converter - Starting")
    logging.info("="*60)
    
    # Convert paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input directory
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        logging.error(f"Input path is not a directory: {input_dir}")
        sys.exit(1)
    
    try:
        # Initialize Gemini
        model = initialize_gemini(args.api_key)
        
        # Process all files
        total, successful, failed = process_directory(input_dir, output_dir, model)
        
        # Summary
        logging.info("\n" + "="*60)
        logging.info("Processing Complete")
        logging.info("="*60)
        logging.info(f"Total files: {total}")
        logging.info(f"Successful: {successful}")
        logging.info(f"Failed: {failed}")
        logging.info(f"Output directory: {output_dir.absolute()}")
        
        if failed > 0:
            sys.exit(1)
        
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.warning("\nProcess interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
