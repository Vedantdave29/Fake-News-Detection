import streamlit as st, joblib, re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Load the saved model and vectorizer
model = joblib.load("models/model.pkl")
tfidf = joblib.load("models/vectorizer.pkl")
stop_words = set(stopwords.words("english"))
lemm = WordNetLemmatizer()

# Same cleaning as step2
def clean(t):
    t = re.sub(r"^.*?\(Reuters\)\s*-?\s*", "", str(t))
    t = re.sub(r"http\S+|www\S+", "", t.lower())
    t = re.sub(r"[^a-z\s]", " ", t)
    return " ".join(lemm.lemmatize(w) for w in t.split()
                    if w not in stop_words and len(w) > 2)

# Page layout
st.title("Fake News Detector")
text = st.text_area("Paste a news headline or article", height=200)

if st.button("Check"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        vec = tfidf.transform([clean(text)])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        if pred == 1:
            st.success(f"REAL NEWS (confidence: {prob[1]*100:.1f}%)")
        else:
            st.error(f"FAKE NEWS (confidence: {prob[0]*100:.1f}%)")
        st.caption("AI prediction based on writing style. Please verify with trusted sources.")