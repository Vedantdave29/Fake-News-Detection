import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load the cleaned data
df = pd.read_csv("data/clean_data.csv").dropna()

# Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

# Turn words into numbers (TF-IDF)
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)   # learn from training data only
X_test_vec = tfidf.transform(X_test)

# Train 4 models and compare them
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Passive Aggressive": PassiveAggressiveClassifier(max_iter=50, random_state=42),
    "Linear SVM": LinearSVC(),
}
for name, m in models.items():
    m.fit(X_train_vec, y_train)
    pred = m.predict(X_test_vec)
    print(f"{name}: Accuracy={accuracy_score(y_test, pred):.4f}  F1={f1_score(y_test, pred):.4f}")

# Show the detailed report for the best model
best = models["Logistic Regression"]
print(classification_report(y_test, best.predict(X_test_vec), target_names=["Fake", "Real"]))

# Save the model and vectorizer for the app
joblib.dump(best, "models/model.pkl")
joblib.dump(tfidf, "models/vectorizer.pkl")
print("Saved model.pkl and vectorizer.pkl")