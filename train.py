import os
import pickle
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

CSV_PATH = "phishingemails/Phishing_Legitimate_emails.csv"

if not os.path.exists(CSV_PATH):
    raise SystemExit(f"Arquivo CSV não encontrado em {CSV_PATH}. Baixe e descompacte o dataset conforme instruções.")

df = pd.read_csv(CSV_PATH, encoding="latin1", engine="python", on_bad_lines="skip")
print("Colunas detectadas no CSV:", df.columns.tolist())

TEXT_COL = None
LABEL_COL = None

for c in df.columns:
    if df[c].dtype == object and df[c].str.len().median() > 20:
        TEXT_COL = c
        break
if TEXT_COL is None:
    TEXT_COL = df.columns[0]

for c in df.columns:
    vals = df[c].dropna().astype(str).str.lower()
    if vals.isin(["phishing","phish","phishing email","phishing_site","1"]).any() or vals.isin(["legitimate","legit","ham","0"]).any():
        LABEL_COL = c
        break

if LABEL_COL is None:
    for c in df.columns:
        uniq = set(df[c].dropna().astype(str).unique())
        if uniq <= {"0","1"}:
            LABEL_COL = c
            break

if LABEL_COL is None:
    raise SystemExit("Não foi possível identificar automaticamente a coluna de label. Abra o CSV e ajuste LABEL_COL manualmente.")

print(f"Usando TEXT_COL = '{TEXT_COL}', LABEL_COL = '{LABEL_COL}'")

df = df[[TEXT_COL, LABEL_COL]].dropna()
df[TEXT_COL] = df[TEXT_COL].astype(str)

def normalize_label(v):
    s = str(v).strip().lower()
    if s in ("1","phish","phishing","phishing email","phishing_site"):
        return 1
    if s in ("0","legit","legitimate","ham","not phishing"):
        return 0
    if "phish" in s:
        return 1
    return 0

df["label_bin"] = df[LABEL_COL].apply(normalize_label)
print("Distribuição de classes:\n", df["label_bin"].value_counts())

X_train, X_test, y_train, y_test = train_test_split(df[TEXT_COL], df["label_bin"], test_size=0.2, random_state=42, stratify=df["label_bin"])

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

print("Iniciando treinamento...")
pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/model_pipeline.pkl"
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)
print(f"Modelo salvo em {MODEL_PATH}")