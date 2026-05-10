import pandas as pd

df = pd.read_csv("tomtom_data/eta_data.csv")
print(df.shape)
print(df["time_label"].value_counts())
print(df.head(3).to_string())
