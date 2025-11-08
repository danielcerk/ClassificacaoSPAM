import os
import re
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "emails.csv")

MODEL_PATH = os.path.join(BASE_DIR, "spam_model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.joblib")


try:

    nltk.data.find("tokenizers/punkt")

except LookupError:

    nltk.download("punkt")

try:

    nltk.data.find("corpora/stopwords")

except LookupError:

    nltk.download("stopwords")


STOPWORDS = set(stopwords.words("english"))
STEMMER = SnowballStemmer("english")


def normalize_text(text: str) -> str:

    if not isinstance(text, str):

        text = str(text)

    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^A-Za-zÀ-ÿ0-9\s]", " ", text)

    tokens = word_tokenize(text, language="english")

    processed = [
        STEMMER.stem(t.lower())
        for t in tokens
        if t.isalpha() and t.lower() not in STOPWORDS and len(t) > 2
    ]

    return " ".join(processed)

def etl_pipeline(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:

    df = df[[text_col, label_col]].dropna()
    df[label_col] = df[label_col].astype(int)

    spam = df[df[label_col] == 1]
    ham = df[df[label_col] == 0]

    if len(spam) < len(ham):

        spam = resample(spam, replace=True, n_samples=len(ham), random_state=42)

    elif len(ham) < len(spam):

        ham = resample(ham, replace=True, n_samples=len(spam), random_state=42)

    df_balanced = pd.concat([spam, ham]).sample(frac=1, random_state=42).reset_index(drop=True)
    df_balanced["text_norm"] = df_balanced[text_col].apply(normalize_text)

    return df_balanced


def preprocessing_and_train(csv_path=CSV_PATH, test_size=0.2, random_state=42):

    if not os.path.exists(csv_path):

        raise FileNotFoundError(f"CSV não encontrado em {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8", engine="python", on_bad_lines="skip")

    text_col = None
    label_col = None

    for c in df.columns:

        if c.lower() in ("texto", "text", "message", "email"):

            text_col = c

        if c.lower() in ("spam", "label", "class"):

            label_col = c

    if text_col is None:

        text_col = df.columns[0]

    if label_col is None:

        label_col = df.columns[-1]

    df_processed = etl_pipeline(df, text_col, label_col)

    X = df_processed["text_norm"].values
    y = df_processed[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)

    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, digits=4)

    print(f"Acurácia no conjunto de teste: {acc:.4f}")
    print("Relatório de classificação:\n", report)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "accuracy": acc,
        "report": report
    }


def spam_detector(content: str):

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        
        raise FileNotFoundError("Modelo ou vetorizer não encontrado. Rode preprocessing_and_train() para treinar e salvar.")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    normalized = normalize_text(content)
    X = vectorizer.transform([normalized])
    pred = model.predict(X)[0]

    prob = {}

    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(X)[0]
        prob = {int(i): float(p) for i, p in enumerate(probs)}

    return {
        "label": int(pred),
        "label_name": "spam" if int(pred) == 1 else "ham",
        "probabilities": prob,
    }


if __name__ == "__main__":

    results = preprocessing_and_train()

    sample_ham = "Subject: request for a quote for corporate website development. Hello, I would like to know the price."
    sample_spam = "Congratulations!!! You have been approved for a loan with no credit check. Click here to claim $5000 now."

    print("\nExemplo HAM:", spam_detector(sample_ham))
    print("Exemplo SPAM:", spam_detector(sample_spam))
