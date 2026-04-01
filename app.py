import os
import sys
import re
import json
import sqlite3
import threading
import webbrowser
import time
from pathlib import Path

from flask import Flask, request, jsonify, send_file, send_from_directory
from gensim.models import Word2Vec
import numpy as np

app = Flask(__name__)

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent  
else:
    BASE_DIR = Path(__file__).parent        

MODEL_PATH   = BASE_DIR / "imessage_word2vec.model"
VECTORS_TSV  = BASE_DIR / "vectors.tsv"
METADATA_TSV = BASE_DIR / "metadata.tsv"
INDEX_HTML   = BASE_DIR / "index.html"
PROJECTOR_DIR = BASE_DIR / "embedding-projector-standalone"
TOP_N        = 10_000
PORT         = 5050


def train_from_imessages() -> Word2Vec:
    import jieba

    db_path = os.path.expanduser("~/Library/Messages/chat.db")
    print("Reading iMessages...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT text FROM message WHERE text IS NOT NULL")
    texts = [row[0] for row in cursor.fetchall()]
    conn.close()
    print(f"  {len(texts):,} messages loaded")

    def clean(text: str) -> list[str]:
        text = re.sub(r"\s", " ", text.strip()).lower()
        tokens = jieba.lcut(text)
        return [t for t in tokens if t.strip()]

    print("Tokenizing...")
    sentences = [clean(t) for t in texts if t.strip()]

    print("Training Word2Vec (this takes a minute)...")
    m = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4, seed=123)
    m.save(str(MODEL_PATH))
    print(f"  Model saved → {MODEL_PATH}")
    return m

if MODEL_PATH.exists():
    print(f"Loading model from {MODEL_PATH}...")
    model = Word2Vec.load(str(MODEL_PATH))
else:
    print("No saved model found — training from iMessages...")
    model = train_from_imessages()

print(f"Ready. Vocabulary: {len(model.wv.key_to_index):,} words")



def export_tsvs() -> None:
    words: list[str]          = list(model.wv.key_to_index.keys())[:TOP_N]
    vectors: list[np.ndarray] = [model.wv.get_vector(w) for w in words]

    print(f"Writing vectors.tsv ({len(words):,} words)...")
    with open(VECTORS_TSV, "w", encoding="utf-8", newline="\n") as f:
        for vec in vectors:
            f.write("\t".join(f"{x:.6f}" for x in vec) + "\n")

    print("Writing metadata.tsv...")
    with open(METADATA_TSV, "w", encoding="utf-8", newline="\n") as f:
        for word in words:
            f.write(word + "\n")

    print("TSVs ready.")


@app.route("/")
def index():
    return send_file(INDEX_HTML)

# Serves the TSV files so the projector iframe can fetch them
@app.route("/vectors.tsv")
def vectors():
    return send_file(VECTORS_TSV, mimetype="text/tab-separated-values")

@app.route("/metadata.tsv")
def metadata():
    return send_file(METADATA_TSV, mimetype="text/tab-separated-values")

# Serves the local projector (cloned from GitHub) to avoid HTTPS/HTTP conflicts
@app.route("/projector/")
@app.route("/projector")
def projector_index():
    return send_file(PROJECTOR_DIR / "index.html")

@app.route("/projector/<path:filename>")
def projector_static(filename):
    return send_from_directory(PROJECTOR_DIR, filename)


def word_ok(w: str) -> bool:
    return w in model.wv.key_to_index

@app.route("/api/ping")
def ping():
    return jsonify({"status": "ok", "vocab_size": len(model.wv.key_to_index)})

@app.route("/api/analogy", methods=["POST"])
def analogy():
    
    d   = request.json
    pos = d.get("positive", [])   # ["king", "woman"]
    neg = d.get("negative", [])   # ["man"]
    missing = [w for w in pos + neg if not word_ok(w)]
    if missing:
        return jsonify({"error": f"Not in vocabulary: {', '.join(missing)}"}), 400
    results = model.wv.most_similar(positive=pos, negative=neg, topn=8)
    return jsonify([{"word": w, "score": round(float(s), 4)} for w, s in results])

@app.route("/api/wordpath", methods=["POST"])
def wordpath():
    d     = request.json
    start = d.get("start", "")
    end   = d.get("end", "")
    steps = min(int(d.get("steps", 6)), 10)
    for w in [start, end]:
        if not word_ok(w):
            return jsonify({"error": f"Not in vocabulary: {w}"}), 400
    path    = [start]
    current = start
    seen    = {start}
    for _ in range(steps):
        neighbors = [w for w, _ in model.wv.most_similar(current, topn=30) if w not in seen]
        if not neighbors:
            break
        current = max(neighbors, key=lambda w: model.wv.similarity(w, end))
        seen.add(current)
        path.append(current)
        if current == end:
            break
    sims = [round(float(model.wv.similarity(path[i], path[i+1])), 4) for i in range(len(path)-1)]
    return jsonify({"path": path, "similarities": sims})

@app.route("/api/sentiment", methods=["POST"])
def sentiment():
    d     = request.json
    words = d.get("words", [])
    pos_a = d.get("positive", "good")
    neg_a = d.get("negative", "bad")
    for w in [pos_a, neg_a]:
        if not word_ok(w):
            return jsonify({"error": f"Anchor not in vocabulary: {w}"}), 400
    results = []
    for word in words:
        if not word_ok(word):
            results.append({"word": word, "error": "not in vocabulary"})
            continue
        ps = float(model.wv.similarity(word, pos_a))
        ns = float(model.wv.similarity(word, neg_a))
        results.append({"word": word, "pos_sim": round(ps,4), "neg_sim": round(ns,4), "score": round(ps-ns,4)})
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return jsonify(results)

@app.route("/api/similar", methods=["POST"])
def similar():
    d    = request.json
    word = d.get("word", "")
    topn = d.get("topn", 12)
    if not word_ok(word):
        return jsonify({"error": f"Not in vocabulary: {word}"}), 400
    results = model.wv.most_similar(word, topn=topn)
    return jsonify([{"word": w, "score": round(float(s), 4)} for w, s in results])


def ensure_projector() -> None:
    if PROJECTOR_DIR.exists():
        return
    print("Downloading embedding projector")
    import subprocess as sp
    sp.run([
        "git", "clone",
        "https://github.com/tensorflow/embedding-projector-standalone",
        str(PROJECTOR_DIR)
    ], check=True)
    print("Projector ready.")

if __name__ == "__main__":
    ensure_projector()
    export_tsvs()

    def open_browser():
        time.sleep(1.0)
        print(f"\nOpening http://localhost:{PORT} ...")
        webbrowser.open(f"http://localhost:{PORT}")

    threading.Thread(target=open_browser, daemon=True).start()

    print(f"\nAll done. Server running at http://localhost:{PORT}")
    print("Close this window to stop the app.\n")
    app.run(port=PORT, debug=False, use_reloader=False)