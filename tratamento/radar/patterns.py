from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from radar import config as cfg

def choose_k(X, k_range=cfg.KMEANS_K_RANGE, sample_size: int = 8000) -> tuple[int, float]:
    n = X.shape[0]
    if n > sample_size:
        idx = np.random.RandomState(cfg.RANDOM_STATE).choice(n, sample_size, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    best_k, best_sil = 10, -1.0
    for k in k_range:
        if Xs.shape[0] <= k:
            continue
        km = MiniBatchKMeans(n_clusters=k, random_state=cfg.RANDOM_STATE, n_init=3, batch_size=2048)
        labels = km.fit_predict(Xs)
        sil = float(silhouette_score(Xs, labels, sample_size=min(3000, len(labels))))
        if sil > best_sil:
            best_sil, best_k = sil, k
    return best_k, best_sil

def run(df: pd.DataFrame | None = None) -> dict[str, Any]:
    if df is None:
        df = pd.read_parquet(cfg.CURATED_PATH)
        df["aberto"] = pd.to_datetime(df["aberto"])

    cfg.OUTPUTS.mkdir(parents=True, exist_ok=True)
    cfg.MODELS.mkdir(parents=True, exist_ok=True)
    print("patterns: TF-IDF + MiniBatchKMeans...")

    templates = df["template"].fillna("").astype(str)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=5000)
    X = vectorizer.fit_transform(templates)

    k, sil = choose_k(X)
    print(f"patterns: k={k} silhouette={sil:.3f}")
    km = MiniBatchKMeans(n_clusters=k, random_state=cfg.RANDOM_STATE, n_init=5, batch_size=4096)
    labels = km.fit_predict(X)
    df = df.copy()
    df["cluster"] = labels

    terms = np.array(vectorizer.get_feature_names_out())
    profiles: dict[str, Any] = {"k": k, "silhouette": round(sil, 4), "clusters": {}}
    cluster_rows = []
    ref = pd.Timestamp(cfg.REF_DATE)
    last_30 = ref - pd.Timedelta(days=30)
    prev_30 = last_30 - pd.Timedelta(days=30)

    for c in range(k):
        mask = df["cluster"] == c
        sub = df[mask]
        center = km.cluster_centers_[c]
        top_idx = center.argsort()[::-1][:8]
        top_terms = terms[top_idx].tolist()
        n = int(mask.sum())
        n_recent = int(((sub["aberto"] > last_30) & (sub["aberto"] <= ref)).sum())
        n_prev = int(((sub["aberto"] > prev_30) & (sub["aberto"] <= last_30)).sum())
        trend = (n_recent - n_prev) / max(n_prev, 1)
        profile = {
            "size": n,
            "top_terms": top_terms,
            "pct_sem_intervencao": round(float(sub["sem_intervencao"].mean()) * 100, 1),
            "pct_kpi_violado": round(float(sub.loc[sub["eh_kpi"], "kpi_violado_flag"].mean()) * 100, 2)
            if sub["eh_kpi"].any()
            else 0.0,
            "top_grupo": sub["grupo_designado"].value_counts().index[0] if len(sub) else None,
            "top_produto": sub["produto"].value_counts().index[0] if len(sub) else None,
            "n_30d": n_recent,
            "n_prev_30d": n_prev,
            "trend_30d": round(float(trend), 3),
        }
        profiles["clusters"][str(c)] = profile
        cluster_rows.append({"cluster": c, **{kk: vv for kk, vv in profile.items() if kk != "top_terms"}, "top_terms": ", ".join(top_terms)})

    clusters_df = df[["numero", "aberto", "template", "template_id", "grupo_designado", "produto", "categoria", "prioridade", "cluster", "sem_intervencao", "eh_kpi", "kpi_violado_flag"]].copy()
    clusters_df.to_parquet(cfg.CLUSTERS_PATH, index=False)
    cfg.CLUSTER_PROFILES_PATH.write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"kmeans": km, "vectorizer": vectorizer, "k": k}, cfg.MODELS / "kmeans.pkl")

    print("patterns: agrupamentos críticos...")
    crit_rows = []
    for keys in [("produto", "categoria", "prioridade"), ("grupo_designado", "prioridade")]:
        g = (
            df.groupby(list(keys))
            .agg(
                n=("numero", "size"),
                n_kpi=("eh_kpi", "sum"),
                n_viol=("kpi_violado_flag", "sum"),
                mediana_duracao_h=("duracao_h", "median"),
                pct_manual=("origem", lambda s: (s == "Manual").mean()),
            )
            .reset_index()
        )
        g = g[g["n"] >= 20]
        g["taxa_violacao"] = g["n_viol"] / g["n_kpi"].clip(lower=1)
        g["dimensao"] = "+".join(keys)
        recent = df[df["aberto"] > last_30]
        counts = recent.groupby(list(keys)).size().rename("n_30d")
        g = g.merge(counts, on=list(keys), how="left").fillna({"n_30d": 0})
        g["score_risco"] = g["taxa_violacao"] * np.log1p(g["n_kpi"]) + 0.01 * g["n_30d"]
        for _, row in g.iterrows():
            crit_rows.append(row.to_dict())
    crit = pd.DataFrame(crit_rows).sort_values("score_risco", ascending=False)
    crit.to_parquet(cfg.CRITICAL_GROUPS_PATH, index=False)

    print("patterns: recorrência IC×template...")
    df_sorted = df.sort_values("aberto")
    rec_rows = []
    for (ic, tpl), grp in df_sorted.groupby(["item_config", "template_id"]):
        if ic == "NA" or len(grp) < 3:
            continue
        times = grp["aberto"]
        n_7 = int((times > ref - pd.Timedelta(days=7)).sum())
        n_30 = int((times > ref - pd.Timedelta(days=30)).sum())
        n_90 = int((times > ref - pd.Timedelta(days=90)).sum())
        diffs = times.diff().dt.total_seconds() / 86400
        rec_rows.append(
            {
                "item_config": ic,
                "template_id": tpl,
                "template": grp["template"].iloc[0],
                "n_total": int(len(grp)),
                "n_7d": n_7,
                "n_30d": n_30,
                "n_90d": n_90,
                "intervalo_medio_d": round(float(diffs.dropna().mean()), 2) if diffs.notna().any() else None,
                "ultima": str(times.max().date()),
                "pct_sem_intervencao": round(float(grp["sem_intervencao"].mean()) * 100, 1),
                "grupo_designado": grp["grupo_designado"].mode().iloc[0] if len(grp) else None,
            }
        )
    rec = pd.DataFrame(rec_rows).sort_values("n_30d", ascending=False)
    rec.to_parquet(cfg.RECURRENCE_PATH, index=False)

    storms = (
        df[df["eh_filho"]]
        .groupby("incidente_pai")
        .size()
        .reset_index(name="n_filhos")
        .sort_values("n_filhos", ascending=False)
    )
    storms.to_parquet(cfg.OUTPUTS / "storms.parquet", index=False)

    summary = {
        "k": k,
        "silhouette": round(sil, 4),
        "n_critical_groups": int(len(crit)),
        "n_recurrence_pairs": int(len(rec)),
        "top_recurrence_30d": int(rec["n_30d"].max()) if len(rec) else 0,
    }
    print(f"patterns: {summary}")
    return summary

if __name__ == "__main__":
    run()
