import os
import sys
import json
import pandas as pd
import numpy as np
from pandas import json_normalize
import re
import argparse

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Normalize JSON park data files into a structured CSV."
)
parser.add_argument(
    "--input-dir",
    "-i",
    default=".",
    help="Directory containing JSON files (default: current directory)"
)
parser.add_argument(
    "--output",
    "-o",
    default="normalized_parks_output.csv",
    help="Output CSV filename (default: normalized_parks_output.csv)"
)
args = parser.parse_args()

# Expand user path (e.g., ~/) and resolve to absolute path
INPUT_DIR = os.path.expanduser(args.input_dir)
OUTPUT_FILE = args.output

# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def flatten_json(json_data):
    """Flattens nested JSON and converts arrays to comma-separated strings."""
    df = json_normalize(json_data)
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x
        )
    return df


def to_tristate(series, patterns):
    """Returns tristate Yes/No/Don't Know using keyword patterns."""
    s = series.str.lower().fillna("")
    escaped = [re.escape(p) for p in patterns]
    joined = "|".join(escaped)
    pattern = r"\b(?:" + joined + r")\b"
    yes_mask = s.str.contains(pattern, regex=True, na=False)
    no_mask = s.str.contains(r"\bno\s+(?:" + joined + r")\b", regex=True, na=False)
    return np.select([yes_mask, no_mask], ["Yes", "No"], default="Don't Know")


# ---------------------------------------------------------------------
# FEATURE MAPS (Keyword sets)
# ---------------------------------------------------------------------

# --- Facilities ---
facilities_map = {
    "has_restrooms": ["restroom", "toilet"],
    "has_picnic_area": ["picnic area", "picnic table", "picnicking"],
    "has_pavilion": ["pavilion", "shade pavilion"],
    "has_playground": ["playground"],
    "has_boat_ramp": ["boat ramp", "boat launch"],
    "has_grills": ["grill", "grilling"],
    "has_showers": ["shower", "bathhouse"],
    "has_visitor_center": ["visitor center", "visitors center", "info booth"],
    "has_scenic_view": ["scenic view", "overlook", "observation tower"],
}

# --- Restrictions ---
restrictions_map = {
    "no_alcohol": ["no alcohol", "no alcoholic"],
    "dogs_allowed": ["dogs allowed", "pets allowed"],
    "dogs_on_leash": ["on-leash", "on leash", "must remain on-leash"],
    "no_fires_or_grills": ["no fires", "no grilling", "no firewood"],
    "carry_in_carry_out": ["carry-in", "carry out", "lug in", "lug out"],
    "no_motorized_vehicles": ["no motorized", "no atv", "no ohv"],
    "no_swimming": ["no swimming", "swimming prohibited"],
    "no_hunting_or_fishing": ["no hunting", "no fishing"],
    "no_drones": ["no drones", "no drone"],
    "service_animals_allowed": ["service animal", "except service animals"],
}

# --- Accessibility ---
accessibility_map = {
    "accessible_restrooms": ["accessible restrooms", "accessible toilet"],
    "accessible_beach": ["accessible beach", "beach mat", "beach wheelchair"],
    "accessible_parking": ["accessible parking"],
    "accessible_trails": ["accessible trail", "accessible hiking"],
    "accessible_camping": ["accessible camping", "accessible campsites"],
    "accessible_picnicking": ["accessible picnicking", "accessible pavilion"],
    "accessible_visitors_center": [
        "accessible visitor center",
        "accessible visitors center",
        "accessible nature center",
    ],
    "accessible_fishing": ["accessible fishing", "accessible pier"],
    "adaptive_recreation_programs": [
        "adaptive recreation",
        "adaptive sports",
        "spaulding adaptive",
    ],
}

# --- Activities ---
activities_map = {
    "hiking": ["hiking", "walk", "trail"],
    "biking": ["biking", "cycling", "bike"],
    "camping": ["camping", "campground"],
    "fishing": ["fishing", "angling"],
    "boating": ["boating", "canoeing", "kayaking", "sailing"],
    "swimming": ["swimming", "swim", "beach"],
    "picnicking": ["picnicking", "picnic"],
    "hunting": ["hunting", "hunt"],
    "horseback_riding": ["horseback", "equestrian"],
    "wildlife_watching": ["bird watching", "nature watching", "wildlife"],
    "winter_sports": ["skiing", "snowmobiling", "ice skating"],
    "rock_climbing": ["rock climbing", "climb"],
    "educational_programs": ["educational", "tours", "guided tour"],
    "events": ["concert", "community event", "triathlon"],
}

# ---------------------------------------------------------------------
# LOAD AND FLATTEN JSON FILES
# ---------------------------------------------------------------------

json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
print(f"Found {len(json_files)} JSON files.")

all_data = []

for file in json_files:
    with open(os.path.join(INPUT_DIR, file), "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            for entry in data:
                all_data.append(flatten_json(entry))
        else:
            all_data.append(flatten_json(data))

if not all_data:
    raise ValueError("No JSON data found.")

combined_df = pd.concat(all_data, ignore_index=True)

# ---------------------------------------------------------------------
# NORMALIZATION PIPELINE
# ---------------------------------------------------------------------

# Clean whitespace
for col in combined_df.columns:
    if combined_df[col].dtype == "object":
        combined_df[col] = combined_df[col].astype(str).str.strip()

# Ensure text columns exist
for required in ["facilities", "restrictions", "accessibility", "activities"]:
    if required not in combined_df.columns:
        combined_df[required] = np.nan

# --- Apply mappings ---
def apply_map(df, source_col, mapping, other_col):
    """Applies tristate mapping and creates an 'other_' text field."""
    for col, keywords in mapping.items():
        df[col] = to_tristate(df[source_col], keywords)
    all_keywords = [kw for sub in mapping.values() for kw in sub]
    mask = ~df[source_col].str.lower().str.contains("|".join(all_keywords), na=False)
    df[other_col] = df[source_col].where(mask, np.nan)
    return df


combined_df = apply_map(combined_df, "facilities", facilities_map, "other_facilities")
combined_df = apply_map(
    combined_df, "restrictions", restrictions_map, "other_restrictions"
)
combined_df = apply_map(
    combined_df, "accessibility", accessibility_map, "other_accessibility_features"
)
combined_df = apply_map(combined_df, "activities", activities_map, "other_activities")

# ---------------------------------------------------------------------
# CREATE METADATA.FEATURES COLUMN
# ---------------------------------------------------------------------

# Collect all boolean flag columns (those with Yes/No/Don't Know values)
flag_columns = []
for col in combined_df.columns:
    if any(col.startswith(prefix) for prefix in ["has_", "no_", "accessible_", "dogs_", "service_", "carry_"]):
        flag_columns.append(col)
    elif col in ["hiking", "biking", "camping", "fishing", "boating", "swimming", 
                 "picnicking", "hunting", "horseback_riding", "wildlife_watching", 
                 "winter_sports", "rock_climbing", "educational_programs", "events"]:
        flag_columns.append(col)

# Create metadata.features column with comma-separated list of "Yes" flags
def get_active_features(row):
    """Returns comma-separated list of column names where value is 'Yes'."""
    active = [col.replace("_", " ") for col in flag_columns if row.get(col) == "Yes"]
    return ", ".join(active) if active else ""

combined_df["metadata.features"] = combined_df.apply(get_active_features, axis=1)

# ---------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------

combined_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
print(f"✅ Normalized dataset created: {OUTPUT_FILE}")
print(f"Rows: {len(combined_df)}, Columns: {len(combined_df.columns)}")

# Optional: print a quick summary
summary = combined_df[
    [c for c in combined_df.columns if any(x in c for x in ["has_", "no_", "accessible_", "hiking"])]
].apply(pd.Series.value_counts)
print("\nQuick summary of flag distributions:\n", summary)
