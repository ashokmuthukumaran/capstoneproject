# app.py
"""
Convenience runner:
1) Ensure CSVs are present (e.g., app_store_reviews.csv).
2) Set: export GEMINI_API_KEY="your-key"
3) python app.py
"""
import os
import json
import subprocess
import sys

CONFIG = "config.json"
"""
If not present, a config.json will be created from config.example.json
"""
if not os.path.exists(CONFIG):
    with open("config.example.json", "r") as f:
        example = json.load(f)
        
    with open(CONFIG, "w") as f:
        json.dump(example, f, indent=2)
    print("Created config.json from example.")

if "GEMINI_API_KEY" not in os.environ:
    print("Warning: GEMINI_API_KEY not set. The pipeline will use rule-based fallbacks.")
print("Running autogen_pipeline.py...")

subprocess.run([sys.executable, "autogen_pipeline.py"], check=True)
print("Done.")
