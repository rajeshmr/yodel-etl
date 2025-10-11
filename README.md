# yodel-etl

A collection of Python scripts for cleaning and normalizing data.

## Scripts

### normalize_parks_json.py

Processes JSON files containing park data and normalizes them into a structured CSV format with standardized features.

**Features:**
- Flattens nested JSON structures
- Extracts and normalizes facilities (restrooms, picnic areas, playgrounds, etc.)
- Identifies restrictions (alcohol, pets, fires, etc.)
- Maps accessibility features (accessible restrooms, trails, parking, etc.)
- Categorizes activities (hiking, biking, camping, fishing, etc.)
- Converts text descriptions into tristate flags (Yes/No/Don't Know)
- Preserves unmatched features in "other_" columns

**Usage:**
```bash
# Process JSON files in current directory
python normalize_parks_json.py

# Specify input directory
python normalize_parks_json.py --input-dir ~/data/yodel/parks_normalized

# Specify both input directory and output file
python normalize_parks_json.py -i ~/data/yodel/parks_normalized -o parks_clean.csv

# Show help
python normalize_parks_json.py --help
```

**Parameters:**
- `--input-dir, -i`: Directory containing JSON files (default: current directory)
- `--output, -o`: Output CSV filename (default: `normalized_parks_output.csv`)

**Input Format:**
- JSON files with park data containing fields like `facilities`, `restrictions`, `accessibility`, and `activities`
- Supports both single objects and arrays of objects

**Output:**
- CSV file with normalized columns for each feature
- Tristate values (Yes/No/Don't Know) for identified features
- Text fields for unmatched features
- `metadata.features` column: comma-separated list of all flag column names with "Yes" values (useful for debugging and verification)

**Dependencies:**
- pandas
- numpy

## Installation

```bash
pip install pandas numpy
```

### test_chat_assistant.py

Tests a chat assistant API by sending questions from a text file and recording responses with retrieval metrics.

**Features:**
- Creates a new session for each question (avoids chat history bias)
- Parses streaming SSE responses
- Extracts reference chunks with similarity scores
- Records results in CSV format for analysis

**Usage:**
```bash
# Test with questions file
python test_chat_assistant.py questions.txt

# Specify output file
python test_chat_assistant.py questions.txt -o results.csv

# Add delay between requests (rate limiting)
python test_chat_assistant.py questions.txt -d 2.0
```

**Input Format:**
- Text file with one question per line

**Output CSV Columns:**
- `question`: The question asked
- `answer`: Complete answer from the assistant
- `status_code`: HTTP status code
- `session_id`: Session identifier
- `total_references`: Number of reference chunks retrieved
- `chunk_ids`: Comma-separated chunk IDs
- `document_names`: Comma-separated document names
- `similarities`: Overall similarity scores
- `vector_similarities`: Vector similarity scores
- `term_similarities`: Term similarity scores

**Dependencies:**
- requests

## Installation

```bash
pip install pandas numpy requests
```

## Contributing

When adding new scripts:
1. Follow the existing code structure
2. Add configuration section at the top
3. Include helper functions with docstrings
4. Document usage in this README
