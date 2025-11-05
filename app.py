from flask import Flask, request, jsonify, send_from_directory
from collections import deque
import time
import re
import os
from urllib.parse import urlparse
import pickle

app = Flask(__name__, static_folder="static", static_url_path="/static")

API_KEYS = {
    "demo-key-123": {"owner": "local-demo"},
}
REQS_PER_WINDOW = 10
WINDOW_SECONDS = 60
_rate_store = {}
URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
SUSPICIOUS_KEYWORDS = [
    "senha", "verifique", "verificação", "atualize", "clique", "urgente",
    "ganhou", "parabéns", "prêmio", "transferir", "confirmar", "verificar",
    "conta suspensa", "limite", "renove", "fatura", "pagamento", "pague"
]
SUSPICIOUS_TLDS = [".ru", ".cn", ".xyz", ".info", ".tk"]

MODEL_PATH = "models/model_pipeline.pkl"
model_pipeline = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model_pipeline = pickle.load(f)
        print("Modelo ML carregado:", MODEL_PATH)
    except Exception as e:
        print("Falha ao carregar modelo:", e)
        model_pipeline = None
else:
    print("Modelo não encontrado. Usando heurísticas.")

def check_api_key(api_key):
    return api_key in API_KEYS

def rate_limit_ok(api_key):
    now = time.time()
    dq = _rate_store.setdefault(api_key, deque())
    while dq and dq[0] <= now - WINDOW_SECONDS:
        dq.popleft()
    if len(dq) >= REQS_PER_WINDOW:
        return False
    dq.append(now)
    return True

def find_urls(text):
    return URL_RE.findall(text)

def suspicious_urls(urls):
    reasons = []
    for u in urls:
        try:
            p = urlparse(u)
            hostname = p.hostname or ""
            for tld in SUSPICIOUS_TLDS:
                if hostname.endswith(tld):
                    reasons.append(f"URL com TLD suspeito: {hostname}")
            if hostname.count(".") >= 4:
                reasons.append(f"Hostname muito longo/possível redirecionamento: {hostname}")
            if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname):
                reasons.append(f"URL usa IP em vez de domínio: {u}")
        except Exception:
            reasons.append(f"URL mal formada: {u}")
    return reasons

def keyword_checks(text):
    text_low = text.lower()
    hits = []
    for kw in SUSPICIOUS_KEYWORDS:
        if kw in text_low:
            hits.append(kw)
    return hits

def suspicious_structure_checks(text):
    reasons = []
    urls = find_urls(text)
    if len(urls) >= 2:
        reasons.append(f"Contém {len(urls)} links — comum em golpes")
    if len(text) > 100 and sum(1 for c in text if c.isupper()) > len(text) * 0.2:
        reasons.append("Uso excessivo de letras maiúsculas")
    if re.search(r"\b(urgente|24 horas|imediatamente|agora|limite)\b", text, re.IGNORECASE):
        reasons.append("Linguagem de urgência/pressão")
    return reasons

def ml_predict(text):
    if not model_pipeline:
        return None
    try:
        pred = model_pipeline.predict([text])[0]
        prob = None
        if hasattr(model_pipeline, "predict_proba"):
            prob = model_pipeline.predict_proba([text])[0].tolist()
        return {"label": int(pred), "probability": prob}
    except Exception:
        return None

def analyze_email_text(text):
    if not text or not text.strip():
        return {"error": "Input vazio ou apenas espaços."}

    ml_res = ml_predict(text)
    if ml_res is not None:
        verdict = "provavelmente legítimo" if ml_res["label"] == 0 else "provavelmente golpe"
        return {
            "verdict": verdict,
            "score": None,
            "reasons": [f"Predição ML: label={ml_res['label']}, prob={ml_res.get('probability')}"],
            "found_urls": find_urls(text),
            "keyword_hits": keyword_checks(text),
            "ml": ml_res
        }

    reasons = []
    score = 0
    kw_hits = keyword_checks(text)
    if kw_hits:
        reasons.append(f"Palavras-chave suspeitas: {', '.join(sorted(set(kw_hits)))}")
        score += min(3, len(kw_hits))
    urls = find_urls(text)
    url_reasons = suspicious_urls(urls)
    reasons.extend(url_reasons)
    score += min(5, len(urls) + len(url_reasons))
    struct_reasons = suspicious_structure_checks(text)
    reasons.extend(struct_reasons)
    score += len(struct_reasons)

    verdict = "provavelmente legítimo"
    if score >= 6:
        verdict = "provavelmente golpe"
    elif score >= 3:
        verdict = "suspeito"

    return {
        "verdict": verdict,
        "score": score,
        "reasons": reasons,
        "found_urls": urls,
        "keyword_hits": kw_hits
    }

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/analyze", methods=["GET"])
def api_analyze():

    data = request.get_json(silent=True)

    text = data.get("text", None)
    if text is None:
        return jsonify({"error": "Campo 'text' ausente."}), 400
    if not isinstance(text, str):
        return jsonify({"error": "Campo 'text' deve ser string."}), 400
    if text.strip() == "":
        return jsonify({"error": "Campo 'text' não pode ser vazio."}), 400

    result = analyze_email_text(text)
    if "error" in result:
        return jsonify({"error": result["error"]}), 400
    return jsonify({"ok": True, "result": result})

if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=5000, debug=True)