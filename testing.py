import pandas as pd

df = pd.read_csv("data/scholarships_parsed.csv")

# See what Qualifications text looks like
print(df["Qualifications"].dropna().sample(20).to_list())