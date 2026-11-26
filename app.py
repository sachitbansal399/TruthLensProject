from flask import Flask, render_template, request
from pathlib import Path
import pickle

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "fake_news_model.pkl"
VECTORIZER_PATH = BASE_DIR / "model" / "tfidf_vectorizer.pkl"

app = Flask(__name__)

model = None
vectorizer = None

def load_model():
    global model, vectorizer
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return False
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return True

load_model()

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    error = None
    text = ""

    if request.method == "POST":
        text = request.form.get("news_text", "").strip()

        if not text:
            error = "Please enter a news headline or article."
        elif model is None or vectorizer is None:
            error = "The model is not trained yet. Run: python train.py"
        else:
            features = vectorizer.transform([text])
            prediction = model.predict(features)[0]

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                confidence = round(float(max(probabilities)) * 100, 2)

            result = "REAL" if prediction == 1 else "FAKE"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        error=error,
        text=text
    )

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
