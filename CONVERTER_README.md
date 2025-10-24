# JSON to Text Converter using Gemini API

This script converts structured JSON files into natural, human-readable English summaries using Google's Gemini API.

## Installation

Install the required dependency:

```bash
pip install google-generativeai
```

## Setup

Set your Gemini API key as an environment variable:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or use `GOOGLE_API_KEY`:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Alternatively, pass the API key directly using the `--api-key` argument.

## Usage

Basic usage:

```bash
python convert_json_to_text.py --input-dir ./input_json --output-dir ./output_text
```

With custom API key:

```bash
python convert_json_to_text.py -i ./input_json -o ./output_text --api-key YOUR_API_KEY
```

With custom log file:

```bash
python convert_json_to_text.py -i ./data -o ./summaries --log-file my_conversion.log
```

## Arguments

- `-i, --input-dir`: Input directory containing JSON files (default: `./input_json`)
- `-o, --output-dir`: Output directory for text files (default: `./output_text`)
- `--api-key`: Gemini API key (optional if set in environment)
- `--log-file`: Log file path (default: `conversion.log`)

## Features

- ✅ Processes all `.json` files in the input directory
- ✅ Converts JSON to natural English summaries via Gemini API
- ✅ Creates output directory automatically
- ✅ Logs progress to both console and file
- ✅ Handles errors gracefully - skips malformed files
- ✅ Rate limiting with 0.5s delay between API calls
- ✅ CLI arguments for flexible configuration
- ✅ Uses `gemini-1.5-flash` model

## Example

Input file: `input_json/winthrop_shore_reservation.json`

Output file: `output_text/winthrop_shore_reservation.txt`

The script will convert structured park data into engaging, Wikipedia-style descriptions that include details about location, hours, activities, facilities, restrictions, accessibility, and nearby attractions.

## Error Handling

- Invalid JSON files are logged and skipped
- API errors are caught and logged
- File I/O errors are handled gracefully
- Process continues even if individual files fail

## Logging

Logs are written to both:
- Console (stdout)
- Log file (default: `conversion.log`)

Each log entry includes timestamp, level, and message.
