import re, nltk, pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
stop_words = set(stopwords.words("english"))
lemm = WordNetLemmatizer()

# Load fake and real news, add labels (0 = fake, 1 = real)
fake = pd.read_csv("data/Fake.csv"); fake["label"] = 0
true = pd.read_csv("data/True.csv"); true["label"] = 1

# Merge, remove duplicates, shuffle
df = pd.concat([fake, true]).drop_duplicates().sample(frac=1, random_state=42)

def clean(t):
    t = re.sub(r"^.*?\(Reuters\)\s*-?\s*", "", str(t))   # remove "(Reuters)" tag
    t = re.sub(r"http\S+|www\S+", "", t.lower())          # lowercase, remove links
    t = re.sub(r"[^a-z\s]", " ", t)                       # keep only letters
    return " ".join(lemm.lemmatize(w) for w in t.split()  # remove common words
                    if w not in stop_words and len(w) > 2)

print("Cleaning... wait 3 to 8 minutes")
df["clean"] = (df["title"].astype(str) + " " + df["text"].astype(str)).apply(clean)

# Save the result
df[["clean", "label"]].to_csv("data/clean_data.csv", index=False)
print("Done!")