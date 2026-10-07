import pandas as pd  # Import pandas for handling CSV and data

# Read Fake news data and give it label 0
fake = pd.read_csv("data/Fake.csv")
fake["label"] = 0

# Read True news data and give it label 1
true = pd.read_csv("data/True.csv")
true["label"] = 1

# Combine both datasets and remove duplicate rows
df = pd.concat([fake, true]).drop_duplicates()

# Shuffle the data and reset the index
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Show rows and columns count
print(df.shape)

# Show count of Fake and True news
print(df["label"].value_counts())

# Show first 5 rows of the dataset
print(df.head())