"""
Script 08: Fast experiment using sentence-transformers + text features.
No LLM forward pass needed. Completes in < 2 minutes.

Features extracted per answer:
  F1: cosine similarity between answer and context (embedding-based)
  F2: answer length (chars)
  F3: word overlap with context (Jaccard)
  F4: unique word ratio in answer

Scientific justification:
  F1 = proxy for faithfulness (similar to NLI-based detectors)
  F2, F3, F4 = lexical features used in SelfCheckGPT-NGram baseline
  Cross-domain test: do these features transfer across QA/Dialogue/Summarization?
"""
import numpy as np, pandas as pd, json, os
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RESULTS = os.path.expanduser("~/.gemini/antigravity/scratch/results")
os.makedirs(RESULTS, exist_ok=True)
SEED = 42; np.random.seed(SEED)
N_PAIRS = 150

# ── Try to load sentence-transformers; fall back to TF-IDF if not available ──
try:
    from sentence_transformers import SentenceTransformer
    print("Loading sentence-transformers (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    USE_EMBEDDINGS = True
    print("✅ Embedding model ready\n")
except ImportError:
    print("sentence-transformers not installed — using TF-IDF cosine similarity\n")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    USE_EMBEDDINGS = False

def cosine_sim(text_a: str, text_b: str) -> float:
    if USE_EMBEDDINGS:
        embs = embedder.encode([text_a, text_b], convert_to_numpy=True)
        num = np.dot(embs[0], embs[1])
        den = np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-9
        return float(num / den)
    else:
        vec = TfidfVectorizer().fit_transform([text_a, text_b])
        return float(cosine_similarity(vec[0], vec[1])[0, 0])

def word_overlap(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def extract_features(context: str, answer: str) -> dict:
    return {
        "cos_sim":        cosine_sim(context, answer),         # F1: faithfulness proxy
        "answer_len":     len(answer),                          # F2: length
        "word_overlap":   word_overlap(context, answer),        # F3: lexical overlap
        "unique_ratio":   len(set(answer.lower().split())) /    # F4: lexical diversity
                          max(len(answer.split()), 1),
    }

# ── Load and extract per domain ───────────────────────────────────────────────
def process_domain(config, n=N_PAIRS):
    path = f"{RESULTS}/{config}_fast_features.csv"
    if os.path.exists(path):
        d = pd.read_csv(path)
        print(f"  ⏭  {config}: loaded cache ({len(d)} rows)")
        return d

    print(f"  Processing {config} ({n} pairs)...", flush=True)
    ds   = load_dataset("pminervini/HaluEval", config)
    data = ds[list(ds.keys())[0]]
    rows = []
    for i in range(min(n, len(data))):
        ex = data[i]
        if i % 50 == 0: print(f"    {i}/{n} ...", flush=True)

        if config == "qa":
            ctx   = ex["knowledge"]
            right = ex["right_answer"]
            hallu = ex["hallucinated_answer"]
        elif config == "dialogue":
            ctx   = ex["knowledge"] + " " + ex["dialogue_history"]
            right = ex["right_response"]
            hallu = ex["hallucinated_response"]
        else:
            ctx   = ex["document"]
            right = ex["right_summary"]
            hallu = ex["hallucinated_summary"]

        for label, ans in [(0, right), (1, hallu)]:
            feats = extract_features(ctx[:500], ans[:300])
            rows.append({"domain": config, "label": label, **feats})

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  ✅ {config}: {len(rows)} rows → {path}")
    return df

# ── Run all domains ───────────────────────────────────────────────────────────
print("=" * 60)
print("EXTRACTING FEATURES (no LLM required)")
print("=" * 60)
dfs = [process_domain(cfg) for cfg in ["qa", "dialogue", "summarization"]]
df  = pd.concat(dfs, ignore_index=True)
domains = df["domain"].unique().tolist()
features = ["cos_sim", "answer_len", "word_overlap", "unique_ratio"]

print(f"\nTotal rows: {len(df)}")
print(df.groupby(["domain","label"]).size().unstack().to_string())

# ── Cross-domain AUROC evaluation ─────────────────────────────────────────────
# cos_sim: higher = more faithful (lower = more likely hallucination) — flip
# word_overlap: higher = more similar to context — flip
flip = {"cos_sim", "word_overlap"}

def prep(X, feats):
    X = X.copy().astype(float)
    for i, f in enumerate(feats):
        if f in flip: X[:, i] = -X[:, i]
    return X

feature_sets = {
    "F1_CosSim (faithful)":  ["cos_sim"],
    "F2_Length":              ["answer_len"],
    "F3_WordOverlap":         ["word_overlap"],
    "F4_UniqueRatio":         ["unique_ratio"],
    "F_ALL (combo)":          features,
}

results = {}
for det, feats in feature_sets.items():
    results[det] = {}
    for tr in domains:
        results[det][tr] = {}
        tr_df = df[df["domain"]==tr]
        X_tr  = prep(tr_df[feats].values, feats)
        y_tr  = tr_df["label"].values
        sc    = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        clf   = LogisticRegression(max_iter=500, random_state=SEED)
        clf.fit(X_tr_s, y_tr)
        for te in domains:
            te_df = df[df["domain"]==te]
            X_te  = prep(te_df[feats].values, feats)
            probs = clf.predict_proba(sc.transform(X_te))[:, 1]
            results[det][tr][te] = round(roc_auc_score(te_df["label"].values, probs), 3)

# ── Print matrix ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CROSS-DOMAIN AUROC MATRIX  (★ = in-domain diagonal)")
print("=" * 70)
for det, mat in results.items():
    print(f"\n  {det}")
    print("    " + "".join(f"  {d:16s}" for d in domains))
    for tr in domains:
        row = f"    {tr:16s}"
        for te in domains:
            v = mat[tr][te]
            row += f"  {v:.3f}{'★' if tr==te else ' ':15s}"
        print(row)

# ── Key finding summary ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("KEY FINDING — In-Domain vs OOD AUROC Gap")
print("=" * 70)
print(f"  {'Detector':25s}  {'In-domain':>10}  {'OOD avg':>8}  {'Gap':>7}  Implication")
print(f"  {'-'*25}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*30}")
for det, mat in results.items():
    in_s, ood_s = [], []
    for tr in domains:
        for te in domains:
            (in_s if tr==te else ood_s).append(mat[tr][te])
    ai, ao = np.mean(in_s), np.mean(ood_s)
    gap = ai - ao
    impl = "✓ transfers well" if gap < 0.05 else ("✗ moderate drop" if gap < 0.15 else "✗✗ large drop")
    print(f"  {det:25s}  {ai:10.3f}  {ao:8.3f}  {gap:+7.3f}  {impl}")

with open(f"{RESULTS}/auroc_fast_results.json","w") as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Results saved → {RESULTS}/auroc_fast_results.json")
print("\n===EXPERIMENT COMPLETE===")
