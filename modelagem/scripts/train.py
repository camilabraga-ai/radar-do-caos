#!/usr/bin/env python3

from __future__ import annotations

import json
import warnings
from pathlib import Path

import holidays
import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "data" / "curated.parquet"
DAILY = ROOT / "outputs" / "daily.parquet"
FEATURES = ROOT / "outputs" / "daily_features.parquet"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

FORECAST_PATH = OUTPUTS / "forecast.parquet"
RISK_PATH = OUTPUTS / "risk.parquet"
IMPORTANCE_PATH = OUTPUTS / "feature_importance.parquet"
CLUSTERS_PATH = OUTPUTS / "clusters.parquet"
METRICS_PATH = OUTPUTS / "metrics.json"

SERIES = ["total", "p2", "p3", "ola_p2", "ola_p3"]
HORIZON = 7
HOLDOUT_DAYS = 28
REGIME_START = pd.Timestamp("2025-09-01")
PROPHET_WINDOW_START = pd.Timestamp("2025-09-01")

LAG_COLS = [
    "y_lag_1",
    "y_lag_2",
    "y_lag_3",
    "y_lag_7",
    "y_lag_14",
    "y_lag_21",
    "y_lag_28",
    "y_roll3_mean",
    "y_roll7_mean",
    "y_roll7_std",
    "y_roll14_mean",
    "y_roll28_mean",
    "y_ema7",
    "y_ema28",
    "y_diff_1",
    "y_diff_7",
    "y_ratio_7",
    "y_vs_roll7",
    "y_vs_roll28",
    "y_dow_mean4",
    "pct_monitoramento",
    "pct_kpi",
    "pct_p2",
    "pct_p3",
    "pct_sem_intervencao",
    "regime_pos_set25",
    "regime_x_mon",
]
CAL_COLS = [
    "tgt_dow",
    "tgt_is_weekend",
    "tgt_month",
    "tgt_day",
    "tgt_weekofyear",
    "tgt_feriado",
    "tgt_feriado_vespera",
    "tgt_is_month_start",
    "tgt_is_month_end",
    "tgt_regime",
    "horizon",
]
FEATURE_COLS = LAG_COLS + CAL_COLS

KMEANS_K = 8
XGB_CLS_END = pd.Timestamp("2025-09-30")

def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "smape": round(_smape(y_true, y_pred), 2),
    }

def _clip0(x: np.ndarray | pd.Series) -> np.ndarray:
    return np.maximum(np.asarray(x, dtype=float), 0.0)

def baseline_predict(hist: pd.DataFrame, future_ds: pd.DatetimeIndex) -> np.ndarray:

    h = hist.set_index("ds")["y"]
    out = []
    for d in future_ds:
        ref = d - pd.Timedelta(days=7)
        out.append(float(h.loc[ref]) if ref in h.index else float(h.tail(7).mean() if len(h) else 0))
    return _clip0(np.array(out))

def _cal_for_dates(dates: pd.DatetimeIndex | pd.Series) -> pd.DataFrame:
    br = holidays.country_holidays("BR")
    ds = pd.to_datetime(pd.Series(dates))
    return pd.DataFrame(
        {
            "tgt_dow": ds.dt.dayofweek.values,
            "tgt_is_weekend": (ds.dt.dayofweek >= 5).astype(int).values,
            "tgt_month": ds.dt.month.values,
            "tgt_day": ds.dt.day.values,
            "tgt_weekofyear": ds.dt.isocalendar().week.astype(int).values,
            "tgt_feriado": ds.dt.date.map(lambda d: int(d in br)).values,
            "tgt_feriado_vespera": (ds + pd.Timedelta(days=1)).dt.date.map(lambda d: int(d in br)).values,
            "tgt_is_month_start": (ds.dt.day <= 3).astype(int).values,
            "tgt_is_month_end": (ds.dt.day >= 28).astype(int).values,
            "tgt_regime": (ds >= REGIME_START).astype(int).values,
        }
    )

def _add_regressors(df: pd.DataFrame) -> pd.DataFrame:
    br = holidays.country_holidays("BR")
    out = df.copy()
    out["ds"] = pd.to_datetime(out["ds"])
    out["regime_pos_set25"] = (out["ds"] >= REGIME_START).astype(float)
    out["feriado_br"] = out["ds"].dt.date.map(lambda d: float(d in br))
    out["is_weekend"] = (out["ds"].dt.dayofweek >= 5).astype(float)
    return out

def prophet_fit_predict(
    train: pd.DataFrame,
    future_ds: pd.DatetimeIndex,
    interval_width: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = _add_regressors(train[["ds", "y"]].copy())
    t = t[t["ds"] >= PROPHET_WINDOW_START]
    if len(t) < 21:
        t = _add_regressors(train[["ds", "y"]].copy())
        t = t[t["ds"] >= pd.Timestamp("2025-07-01")]

    m = Prophet(
        interval_width=interval_width,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.25,
        seasonality_prior_scale=10.0,
        seasonality_mode="multiplicative",
    )
    m.add_regressor("regime_pos_set25")
    m.add_regressor("feriado_br")
    m.add_regressor("is_weekend")
    m.fit(t)

    future = pd.DataFrame({"ds": pd.to_datetime(list(t["ds"]) + list(future_ds))})
    future = future.drop_duplicates("ds").sort_values("ds")
    future = _add_regressors(future)
    fc = m.predict(future)
    pred = fc[fc["ds"].isin(future_ds)].sort_values("ds")
    return _clip0(pred["yhat"].values), _clip0(pred["yhat_lower"].values), _clip0(pred["yhat_upper"].values)

def _build_xgb_frame(feats: pd.DataFrame, train_end: pd.Timestamp, serie: str) -> pd.DataFrame:

    s = feats[feats["serie"] == serie].sort_values("ds").copy()
    s["ds"] = pd.to_datetime(s["ds"])
    s = s[s["ds"] >= REGIME_START]
    y_by_ds = s.set_index("ds")["y"]
    rows = []
    for h in range(1, HORIZON + 1):
        w = s.copy()
        tgt_ds = w["ds"] + pd.Timedelta(days=h)
        seasonal = tgt_ds.map(lambda d: float(y_by_ds.get(d - pd.Timedelta(days=7), np.nan)))
        w["seasonal_anchor"] = seasonal.values
        w["target"] = w["y"].shift(-h) - w["seasonal_anchor"]
        w["horizon"] = h
        cal = _cal_for_dates(tgt_ds)
        for c in cal.columns:
            w[c] = cal[c].values
        w = w.dropna(subset=["target", "seasonal_anchor"] + LAG_COLS)
        w = w[w["ds"] + pd.Timedelta(days=h) <= train_end]
        rows.append(w)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def _xgb_fit(frame: pd.DataFrame) -> XGBRegressor:
    clf = XGBRegressor(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        objective="reg:squarederror",
    )
    clf.fit(frame[FEATURE_COLS], frame["target"].astype(float))
    return clf

def xgb_predict_horizon(
    feats: pd.DataFrame,
    serie: str,
    as_of: pd.Timestamp,
    future_ds: pd.DatetimeIndex,
    model: XGBRegressor,
) -> np.ndarray:
    s = feats[feats["serie"] == serie].sort_values("ds")
    y_by_ds = s.set_index("ds")["y"]
    row = s[s["ds"] == as_of]
    if row.empty or row[LAG_COLS].isna().any(axis=1).iloc[0]:
        row = s.dropna(subset=LAG_COLS).tail(1)
    if row.empty:
        return np.zeros(len(future_ds))

    preds = []
    for i, d in enumerate(future_ds):
        h = i + 1
        anchor = float(y_by_ds.get(d - pd.Timedelta(days=7), row["y_roll7_mean"].iloc[0] or 0))
        base = row[LAG_COLS].iloc[0].to_dict()
        cal = _cal_for_dates(pd.DatetimeIndex([d])).iloc[0].to_dict()
        x = {**base, **cal, "horizon": h}
        X = pd.DataFrame([x])[FEATURE_COLS]
        resid = float(model.predict(X)[0])
        preds.append(anchor + resid)
    return _clip0(np.array(preds))

def evaluate_serie(feats: pd.DataFrame, serie: str) -> tuple[str, dict]:
    s = feats[feats["serie"] == serie].sort_values("ds").copy()
    s["ds"] = pd.to_datetime(s["ds"])
    last = s["ds"].max()
    holdout_start = last - pd.Timedelta(days=HOLDOUT_DAYS - 1)
    train = s[s["ds"] < holdout_start]
    hold = s[(s["ds"] >= holdout_start) & (s["ds"] <= last)]
    if train.empty or len(hold) < HORIZON:
        raise RuntimeError(f"{serie}: holdout insuficiente")

    hold_ds = pd.DatetimeIndex(hold["ds"].head(HORIZON))
    y_true = hold["y"].head(HORIZON).values.astype(float)
    train_end = holdout_start - pd.Timedelta(days=1)
    as_of = train["ds"].max()

    bl = baseline_predict(train[["ds", "y"]], hold_ds)
    ph, _, _ = prophet_fit_predict(train[["ds", "y"]], hold_ds)

    frame = _build_xgb_frame(feats, train_end, serie)
    if frame.empty:
        xg = np.full(HORIZON, float(train["y"].tail(7).mean()))
    else:
        model = _xgb_fit(frame)
        xg = xgb_predict_horizon(feats, serie, as_of, hold_ds, model)

    all_scores = {
        "baseline_sazonal_ref": {**_score(y_true, bl), "mae_d1": _score(y_true[:1], bl[:1])["mae"]},
        "prophet_regimes": {},
        "xgboost_fe": {},
    }
    for name, pred in (("prophet_regimes", ph), ("xgboost_fe", xg)):
        d1 = _score(y_true[:1], pred[:1])
        d7 = _score(y_true, pred)
        all_scores[name] = {"d1": d1, "d1_d7": d7, "mae_d1": d1["mae"]}

    all_scores["baseline_sazonal_ref"] = {
        "d1": _score(y_true[:1], bl[:1]),
        "d1_d7": _score(y_true, bl),
        "mae_d1": _score(y_true[:1], bl[:1])["mae"],
    }

    eligible = {k: v for k, v in all_scores.items() if k in ("prophet_regimes", "xgboost_fe")}
    champion = min(eligible, key=lambda k: eligible[k]["mae_d1"])
    return champion, all_scores

def final_forecast(feats: pd.DataFrame, serie: str, champion: str) -> pd.DataFrame:
    s = feats[feats["serie"] == serie].sort_values("ds").copy()
    s["ds"] = pd.to_datetime(s["ds"])
    last = s["ds"].max()
    future_ds = pd.date_range(last + pd.Timedelta(days=1), periods=HORIZON, freq="D")

    if champion == "prophet_regimes":
        yhat, lo, up = prophet_fit_predict(s[["ds", "y"]], future_ds)
    else:
        frame = _build_xgb_frame(feats, last, serie)
        model = _xgb_fit(frame)
        joblib.dump({"model": model, "features": FEATURE_COLS, "mode": "resid_vs_seasonal"}, MODELS / f"xgb_vol_{serie}.pkl")
        yhat = xgb_predict_horizon(feats, serie, last, future_ds, model)
        std = float(s["y_roll7_std"].iloc[-1] or s["y"].tail(7).std() or 1)
        lo, up = _clip0(yhat - 1.28 * std), yhat + 1.28 * std

    hist = s[["ds", "y"]].copy()
    hist["yhat"] = hist["y"]
    hist["yhat_lower"] = hist["y"]
    hist["yhat_upper"] = hist["y"]
    hist["is_forecast"] = False
    hist["serie"] = serie

    fut = pd.DataFrame(
        {
            "serie": serie,
            "ds": future_ds,
            "y": np.nan,
            "yhat": yhat,
            "yhat_lower": lo,
            "yhat_upper": up,
            "is_forecast": True,
        }
    )
    return pd.concat([hist, fut], ignore_index=True)

def train_risk_and_clusters(df: pd.DataFrame) -> dict:
    kpi = df[df["eh_kpi"]].copy()
    kpi = kpi.dropna(subset=["prioridade_num", "hora", "dow"])
    kpi["grupo_designado"] = kpi["grupo_designado"].fillna("Desconhecido")
    kpi["origem"] = kpi["origem"].fillna("Desconhecido")
    encoders = {}
    for col, name in [("grupo_designado", "grupo_enc"), ("origem", "origem_enc")]:
        le = LabelEncoder()
        kpi[name] = le.fit_transform(kpi[col].astype(str))
        encoders[col] = le
    feats = ["prioridade_num", "grupo_enc", "origem_enc", "hora", "dow"]
    feat_labels = {
        "prioridade_num": "Prioridade",
        "grupo_enc": "Grupo designado",
        "origem_enc": "Origem (manual/monitoramento)",
        "hora": "Horário de abertura",
        "dow": "Dia da semana",
    }
    train = kpi[kpi["aberto"] <= XGB_CLS_END]
    test = kpi[kpi["aberto"] > XGB_CLS_END]
    y_train = train["kpi_violado_flag"].astype(int)
    y_test = test["kpi_violado_flag"].astype(int)
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    spw = (n_neg / n_pos) if n_pos else 1.0
    clf = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    )
    clf.fit(train[feats], y_train)
    proba = clf.predict_proba(test[feats])[:, 1]
    pred = (proba >= 0.5).astype(int)
    f1 = float(f1_score(y_test, pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_test, proba))
    except ValueError:
        auc = float("nan")
    scores = test[["numero", "aberto", "prioridade", "grupo_designado", "origem", "kpi_violado_flag"]].copy()
    scores["score"] = proba
    scores.sort_values("score", ascending=False).to_parquet(RISK_PATH, index=False)
    pd.DataFrame(
        {"feature": feats, "label": [feat_labels[f] for f in feats], "importance": clf.feature_importances_}
    ).sort_values("importance", ascending=False).to_parquet(IMPORTANCE_PATH, index=False)
    joblib.dump({"model": clf, "encoders": encoders, "features": feats}, MODELS / "xgboost.pkl")

    work = df[df["template"].astype(str).str.len() >= 4].copy()
    counts = work.groupby("template").agg(
        n=("numero", "size"),
        n_ola=("kpi_violado_flag", "sum"),
        top_grupo=("grupo_designado", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
        top_produto=("produto", lambda s: s.dropna().mode().iloc[0] if s.dropna().size else ""),
    )
    templates = counts.index.tolist()
    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=1500)
    X = vec.fit_transform(templates)
    k = min(KMEANS_K, max(2, X.shape[0] // 50))
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10, batch_size=1024)
    labels = km.fit_predict(X)
    sample_n = min(4000, X.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X.shape[0], size=sample_n, replace=False)
    sil = float(silhouette_score(X[idx], labels[idx])) if sample_n > k else float("nan")
    terms = np.array(vec.get_feature_names_out())
    rows = []
    for i in range(k):
        center = km.cluster_centers_[i]
        top = terms[center.argsort()[::-1][:5]]
        mask = labels == i
        sub = counts.iloc[mask]
        rows.append(
            {
                "cluster": i,
                "n": int(sub["n"].sum()),
                "n_templates": int(mask.sum()),
                "termos": ", ".join(top.tolist()),
                "pct_ola": round(float(sub["n_ola"].sum() / sub["n"].sum() * 100), 2) if sub["n"].sum() else 0.0,
                "top_grupo": sub.sort_values("n", ascending=False)["top_grupo"].iloc[0] if len(sub) else "",
                "top_produto": sub.sort_values("n", ascending=False)["top_produto"].iloc[0] if len(sub) else "",
                "top_template": sub["n"].idxmax() if len(sub) else "",
            }
        )
    pd.DataFrame(rows).sort_values("n", ascending=False).to_parquet(CLUSTERS_PATH, index=False)
    joblib.dump({"kmeans": km, "vectorizer": vec, "k": k}, MODELS / "kmeans.pkl")
    return {
        "xgboost": {
            "f1": round(f1, 4),
            "auc": None if auc != auc else round(auc, 4),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        },
        "kmeans": {"k": k, "silhouette": None if sil != sil else round(sil, 4), "n_templates": int(len(templates))},
    }

def main() -> None:
    for p in (CURATED, DAILY, FEATURES):
        if not p.exists():
            raise FileNotFoundError(f"Rode prepare.py primeiro. Ausente: {p}")
    MODELS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    print("Lendo artefatos...")
    df = pd.read_parquet(CURATED)
    df["aberto"] = pd.to_datetime(df["aberto"])
    feats = pd.read_parquet(FEATURES)
    feats["ds"] = pd.to_datetime(feats["ds"])

    missing = [c for c in LAG_COLS if c not in feats.columns]
    if missing:
        raise RuntimeError(f"Features ausentes (rode prepare.py): {missing}")

    forecast_frames = []
    prophet_block: dict = {}
    champions: dict = {}

    for serie in SERIES:
        print(f"Avaliando {serie}...")
        champ, scores = evaluate_serie(feats, serie)
        champions[serie] = champ
        print(f"  campeão={champ}  mae_d1={scores[champ]['mae_d1']}  (baseline_ref={scores['baseline_sazonal_ref']['mae_d1']})")
        prophet_block[serie] = {
            "champion": champ,
            "holdout_candidates": scores,
        }
        c = scores[champ]["d1_d7"]
        prophet_block[serie]["rmse"] = c["rmse"]
        prophet_block[serie]["mape"] = c["smape"]
        prophet_block[serie]["smape"] = c["smape"]
        prophet_block[serie]["mae"] = c["mae"]

        print(f"  previsao final ({champ})...")
        fc = final_forecast(feats, serie, champ)
        extra = fc[fc["is_forecast"]]
        if not extra.empty:
            prophet_block[serie]["d1"] = round(float(extra.iloc[0]["yhat"]), 1)
            prophet_block[serie]["d7"] = round(float(extra["yhat"].sum()), 1)
            peak = extra.loc[extra["yhat"].idxmax()]
            prophet_block[serie]["peak_ds"] = str(pd.Timestamp(peak["ds"]).date())
            prophet_block[serie]["peak_yhat"] = round(float(peak["yhat"]), 1)
            prev = fc[(~fc["is_forecast"])].tail(7)["y"].values
            corr = float(np.corrcoef(prev, extra["yhat"].values)[0, 1]) if len(prev) == 7 else None
            prophet_block[serie]["corr_vs_prev_week"] = None if corr != corr else round(corr, 3)
        forecast_frames.append(fc)

    pd.concat(forecast_frames, ignore_index=True).to_parquet(FORECAST_PATH, index=False)

    print("XGBoost risco + KMeans...")
    extra_m = train_risk_and_clusters(df)

    metrics = {
        "version": "ml-v2.1",
        "split": {
            "type": "holdout_last_n_days",
            "holdout_days": HOLDOUT_DAYS,
            "final_train_end": "last_day_of_base",
            "horizon_days": HORIZON,
            "prophet_window_start": str(PROPHET_WINDOW_START.date()),
        },
        "feature_cols": FEATURE_COLS,
        "champions": champions,
        "prophet": prophet_block,
        "xgboost": extra_m["xgboost"],
        "kmeans": extra_m["kmeans"],
        "n_rows": int(len(df)),
        "v1_reference": {"total_mape": 38.83, "total_rmse": 364.33, "total_d1": 563.7},
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "champions": champions,
                "d1": {s: prophet_block[s].get("d1") for s in SERIES},
                "corr_vs_prev_week": {s: prophet_block[s].get("corr_vs_prev_week") for s in SERIES},
            },
            indent=2,
        )
    )
    print(f"Métricas: {METRICS_PATH}")

if __name__ == "__main__":
    main()
