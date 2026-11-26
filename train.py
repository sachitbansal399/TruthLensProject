from pathlib import Path
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

TRUE_FILE = DATA_DIR / "True.csv"
FAKE_FILE = DATA_DIR / "Fake.csv"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

if not TRUE_FILE.exists() or not FAKE_FILE.exists():
    raise FileNotFoundError(
        "\nDataset files not found.\n"
        "Put True.csv and Fake.csv inside the data folder.\n"
        "Then run: python train.py\n"
    )

true_df = pd.read_csv(TRUE_FILE)
fake_df = pd.read_csv(FAKE_FILE)

required = {"text"}
if not required.issubset(true_df.columns) or not required.issubset(fake_df.columns):
    raise ValueError(
        "The CSV files must contain a 'text' column. "
        "The standard ISOT files also contain 'title', 'subject' and 'date'."
    )

def prepare(df, label):
    title = df["title"].fillna("") if "title" in df.columns else ""
    body = df["text"].fillna("")
    combined = title.astype(str) + " " + body.astype(str)
    return pd.DataFrame({"text": combined, "label": label})

data = pd.concat([
    prepare(true_df, 1),
    prepare(fake_df, 0)
], ignore_index=True)

data["text"] = data["text"].map(clean_text)
data = data[data["text"].str.len() > 20]
data = data.drop_duplicates(subset="text")

X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.20,
    random_state=42,
    stratify=data["label"]
)

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)

print("\nTraining complete!")
print(f"Articles used: {len(data):,}")
print(f"Test accuracy: {accuracy * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, predictions, target_names=["FAKE", "REAL"]))

with open(MODEL_DIR / "fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open(MODEL_DIR / "tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nSaved:")
print(MODEL_DIR / "fake_news_model.pkl")
print(MODEL_DIR / "tfidf_vectorizer.pkl")
print("\nNow run: python app.py")
