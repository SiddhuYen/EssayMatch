import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_CSV = "data/scholarships_parsed.csv"
TAXONOMY_JSON = "data/scholarship_taxonomy.json"
OUTPUT_CSV = "data/scholarship_parsed_classified.csv"

MODEL = "gpt-4o-mini"
BATCH_SIZE = 25


def clean(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


with open(TAXONOMY_JSON, "r", encoding="utf-8") as f:
    taxonomy = json.load(f)

df = pd.read_csv(INPUT_CSV)

required_cols = ["Purpose", "Qualifications", "Criteria", "Focus", "To Apply", "url"]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

rows = []

for idx, row in df.iterrows():
    rows.append({
        "row_id": int(idx),
        "Purpose": clean(row["Purpose"]),
        "Qualifications": clean(row["Qualifications"]),
        "Criteria": clean(row["Criteria"]),
        "Focus": clean(row["Focus"]),
        "To Apply": clean(row["To Apply"]),
        "url": clean(row["url"]),
    })


def classify_batch(batch):
    prompt = f"""
You are classifying scholarships into a fixed taxonomy.

Use ONLY the taxonomy provided. Do not invent new categories.

For each scholarship, return:
- row_id
- best_path
- secondary_path
- third_path
- essay_themes
- eligibility_tags
- hard_restrictions
- summary
- confidence

Path format must be:
Parent > Subcategory > Leaf Category

If no good secondary or third path exists, use an empty string.

Use eligibility_tags only from the taxonomy eligibility list.

Taxonomy:
{json.dumps(taxonomy, ensure_ascii=False)}

Scholarships:
{json.dumps(batch, ensure_ascii=False)}

Return JSON only in this exact structure:
{{
  "classified": [
    {{
      "row_id": 0,
      "best_path": "STEM / Technology / Healthcare > Engineering > Biomedical Engineering",
      "secondary_path": "Leadership / Service / Community Impact > Leadership > Project Leadership",
      "third_path": "",
      "essay_themes": ["Technical / Academic Passion", "Leadership / Initiative"],
      "eligibility_tags": ["undergraduate", "specific_major_required"],
      "hard_restrictions": "Must be pursuing engineering.",
      "summary": "Supports engineering students with leadership and technical experience.",
      "confidence": 0.92
    }}
  ]
}}
"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    data = json.loads(response.choices[0].message.content)
    return data.get("classified", [])


classified_by_id = {}

for start in range(0, len(rows), BATCH_SIZE):
    batch = rows[start:start + BATCH_SIZE]
    print(f"Classifying rows {start} to {start + len(batch) - 1}...")

    try:
        classified = classify_batch(batch)

        for item in classified:
            classified_by_id[int(item["row_id"])] = item

    except Exception as e:
        print(f"Batch failed at row {start}: {e}")

    time.sleep(0.5)


output_rows = []

for idx, row in df.iterrows():
    item = classified_by_id.get(int(idx), {})

    output_rows.append({
        **row.to_dict(),
        "best_path": item.get("best_path", ""),
        "secondary_path": item.get("secondary_path", ""),
        "third_path": item.get("third_path", ""),
        "essay_themes": json.dumps(item.get("essay_themes", []), ensure_ascii=False),
        "eligibility_tags": json.dumps(item.get("eligibility_tags", []), ensure_ascii=False),
        "hard_restrictions": item.get("hard_restrictions", ""),
        "taxonomy_summary": item.get("summary", ""),
        "taxonomy_confidence": item.get("confidence", ""),
    })

out_df = pd.DataFrame(output_rows)
out_df.to_csv(OUTPUT_CSV, index=False)

print(f"Done. Saved to {OUTPUT_CSV}")
print(f"Classified {len(classified_by_id)} / {len(df)} rows.")