"""Script 07-local: Cross-domain AUROC evaluation from local feature files."""
import numpy as np, pandas as pd, json, os
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RESULTS = os.path.expanduser("~/.gemini/antigravity/scratch/results")
SEED    = 42

print(f"Loading features from {RESULTS} ...\n")
dfs = []
for domain in ["qa", "dialogue", "summarization"]:
    for suffix in ["_wb_features", "_features"]:
        path = f"{RESULTS}/{domain}{suffix}.csv"
        if os.path.exists(path):
            d = pd.read_csv(path)
            d["domain"] = domain
            dfs.append(d)
            print(f"  ✅ {domain}: {len(d)} rows")
            break
    else:
        print(f"  ❌ {domain}: missing — run 06_local_wb_features.py first")

if len(dfs) < 2:
    print("\nNeed at least 2 domains. Aborting."); exit(1)

df = pd.concat(dfs, ignore_index=True)
domains = df["domain"].unique().tolist()
wb_cols = [c for c in ["mean_logprob","perplexity","mean_entropy"] if c in df.columns]

feature_sets = {}
if "mean_logprob" in wb_cols:
    feature_sets["B1_LogProb (WB)"]    = ["mean_logprob"]
if "perplexity" in wb_cols:
    feature_sets["B2_Perplexity (WB)"] = ["perplexity"]
if "mean_entropy" in wb_cols:
    feature_sets["B3_Entropy (WB)"]    = ["mean_entropy"]
if len(wb_cols) >= 2:
    feature_sets["B4_WB_combo"]        = wb_cols

flip = {"mean_logprob", "consistency"}
def prep(X, feats):
    X = X.copy().astype(float)
    for i, f in enumerate(feats):
        if f in flip: X[:, i] = -X[:, i]
    return X

results = {}
for det, feats in feature_sets.items():
    results[det] = {}
    for tr in domains:
        results[det][tr] = {}
        tr_df = df[df["domain"]==tr]
        X_tr  = prep(tr_df[feats].values, feats)
        y_tr  = tr_df["label"].values
        sc    = StandardScaler(); X_tr_s = sc.fit_transform(X_tr)
        clf   = LogisticRegression(max_iter=1000, random_state=SEED)
        clf.fit(X_tr_s, y_tr)
        for te in domains:
            te_df = df[df["domain"]==te]
            X_te  = prep(te_df[feats].values, feats)
            probs = clf.predict_proba(sc.transform(X_te))[:, 1]
            results[det][tr][te] = round(roc_auc_score(te_df["label"].values, probs), 3)

print("\n" + "="*70)
print("CROSS-DOMAIN AUROC MATRIX  (★ = in-domain / diagonal)")
print("="*70)
for det, mat in results.items():
    print(f"\n  {det}")
    print("    " + "".join(f"  {d:16s}" for d in domains))
    for tr in domains:
        row = f"    {tr:16s}"
        for te in domains:
            v = mat[tr][te]
            row += f"  {v:.3f}{'★' if tr==te else ' ':15s}"
        print(row)

print("\n" + "="*70)
print("SUMMARY: In-Domain vs OOD Gap  ← KEY RESEARCH FINDING")
print("="*70)
print(f"  {'Detector':22s}  {'In-domain':>10}  {'OOD avg':>8}  {'Gap':>7}")
print(f"  {'-'*22}  {'-'*10}  {'-'*8}  {'-'*7}")
for det, mat in results.items():
    in_s, ood_s = [], []
    for tr in domains:
        for te in domains:
            (in_s if tr==te else ood_s).append(mat[tr][te])
    ai, ao = np.mean(in_s), np.mean(ood_s)
    print(f"  {det:22s}  {ai:10.3f}  {ao:8.3f}  {ai-ao:+7.3f}")

with open(f"{RESULTS}/auroc_results.json","w") as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Saved → {RESULTS}/auroc_results.json")
print("\n===PASTE THIS OUTPUT TO SUPERVISOR===")
