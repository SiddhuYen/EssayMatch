import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CSV_PATH = "data/scholarships_parsed.csv"
OUT_PATH = "data/scholarship_taxonomy.json"

df = pd.read_csv(CSV_PATH)

def clean(v):
    if pd.isna(v):
        return ""
    return str(v).strip()

# Use a sample first so this is cheap.
# Increase to 500 later if needed.
sample = df.sample(min(500, len(df)), random_state=42)

scholarships = []

for i, row in sample.iterrows():
    scholarships.append({
        "id": int(i),
        "purpose": clean(row.get("Purpose", "")),
        "qualifications": clean(row.get("Qualifications", "")),
        "url": clean(row.get("url", "")),
    })

prompt = f"""
You are designing a scholarship matching taxonomy.

I have a dataset of scholarships. Create a reusable category tree that can classify scholarships and student essays.

Requirements:
- Create 8 to 12 parent categories.
- Each parent should have 3 to 8 subcategories.
- Categories should be useful for matching essays to scholarships.
- Avoid categories that are too narrow.
- Include categories for eligibility-heavy scholarships like local/state-specific, identity-specific, career-specific, and financial need.
- Include STEM, healthcare, service, leadership, adversity, creativity, and general education if appropriate.
- Return JSON only.

Scholarship sample:
{json.dumps(scholarships, ensure_ascii=False)[:60000]}

Return this exact JSON structure:

{{
  "parent_categories": [
    {{
      "name": "STEM / Engineering",
      "description": "Scholarships focused on science, engineering, technology, math, innovation, or technical fields.",
      "subcategories": [
        {{
          "name": "Biomedical Engineering",
          "description": "Scholarships related to bioengineering, medical devices, computational biology, or health-focused engineering."
        }}
      ]
    }}
  ],
  "routing_rules": [
    {{
      "rule": "If a scholarship is only available to a specific city, state, school, organization member, gender, ethnicity, or career stage, preserve that as an eligibility tag even if the topic category is broader."
    }}
  ],
  "eligibility_tags": [
    "undergraduate",
    "graduate_only",
    "high_school_senior",
    "state_specific",
    "city_specific",
    "school_specific",
    "member_only",
    "identity_specific",
    "women_only",
    "financial_need",
    "citizenship_required"
  ]
}}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.2,
    response_format={"type": "json_object"},
    messages=[
        {"role": "user", "content": prompt}
    ],
)

taxonomy = json.loads(response.choices[0].message.content)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(taxonomy, f, indent=2, ensure_ascii=False)

print(f"Saved taxonomy to {OUT_PATH}")
print(json.dumps(taxonomy, indent=2, ensure_ascii=False))