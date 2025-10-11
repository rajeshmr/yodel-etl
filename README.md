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

**Dependencies:**
- pandas
- numpy

## Installation

```bash
pip install pandas numpy
```

## Contributing

When adding new scripts:
1. Follow the existing code structure
2. Add configuration section at the top
3. Include helper functions with docstrings
4. Document usage in this README
