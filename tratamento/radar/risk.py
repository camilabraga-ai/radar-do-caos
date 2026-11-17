from __future__ import annotations

import json
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from xgboost import XGBClassifier

from radar import config as cfg
from radar.features import RISK_FEATURE_COLS

warnings.filterwarnings("ignore")

def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if len(scores) == 0:
        return float("nan")
    k = min(k, len(scores))
    idx = np.argsort(scores)[::-1][:k]
    return float(y_true[idx].mean())

def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    total_pos = y_true.sum()
    if total_pos == 0:
        return float("nan")
    k = min(k, len(scores))
    idx = np.argsort(scores)[::-1][:k]
    return float(y_true[idx].sum() / total_pos)

def lift_at_pct(y_true: np.ndarray, scores: np.ndarray, pct: float = 0.05) -> float:
    k = max(1, int(len(scores) * pct))
    base = y_true.mean()
    if base == 0:
        return float("nan")
    return precision_at_k(y_true, scores, k) / base

def severity_from_scores(scores: np.ndarray) -> list[str]:
    q98 = np.quantile(scores, 0.98)
    q90 = np.quantile(scores, 0.90)
    q75 = np.quantile(scores, 0.75)
    out = []
    for s in scores:
        if s >= q98:
            out.append("Crítico")
        elif s >= q90:
            out.append("Alto")
        elif s >= q75:
            out.append("Médio")
        else:
            out.append("Baixo")
    return out

def shap_top_factors(
    model: XGBClassifier,
    X: pd.DataFrame,
    feature_names: list[str],
    n_top: int = 3,
) -> tuple[list[str], dict[str, float]]:
    """SHAP TreeExplainer — top factors por linha + importância global média abs."""
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        sample = X if len(X) <= 2000 else X.sample(2000, random_state=cfg.RANDOM_STATE)
        sv = explainer.shap_values(sample)
        if isinstance(sv, list):
            sv = sv[1]
        global_imp = {
            feature_names[i]: float(np.abs(sv[:, i]).mean())
            for i in range(len(feature_names))
        }
        sv_full = explainer.shap_values(X)
        if isinstance(sv_full, list):
            sv_full = sv_full[1]
        texts = []
        for i in range(len(X)):
            row = sv_full[i]
            top_idx = np.argsort(np.abs(row))[::-1][:n_top]
            parts = []
            for j in top_idx:
                sign = "+" if row[j] >= 0 else "−"
                parts.append(f"{feature_names[j]} ({sign})")
            texts.append(", ".join(parts))
        return texts, global_imp
    except Exception as e:
        print(f"  SHAP indisponível ({e}); usando feature_importances_")
        imp = model.feature_importances_
        global_imp = {feature_names[i]: float(imp[i]) for i in range(len(feature_names))}
        top_global = sorted(global_imp, key=global_imp.get, reverse=True)[:n_top]
        texts = [", ".join(f"{f} (~)" for f in top_global)] * len(X)
        return texts, global_imp

def friendly_factor_names(text: str) -> str:
    mapping = {
        "prioridade_num": "prioridade",
        "grupo_enc": "grupo",
        "produto_enc": "produto",
        "categoria_enc": "categoria",
        "origem_enc": "origem",
        "hora": "hora",
        "dow": "dia_semana",
        "fim_de_semana": "fim_de_semana",
        "horario_comercial": "horário_comercial",
        "mes": "mês",
        "eh_filho": "incidente_filho",
        "feriado_br": "feriado",
        "regime_pos_set25": "regime_pós_set/25",
        "template_freq_30d": "freq_template_30d",
        "ic_count_7d": "recorrência_IC_7d",
        "ic_count_30d": "recorrência_IC_30d",
        "grupo_viol_rate": "taxa_violação_grupo",
        "ic_viol_rate": "taxa_violação_IC",
        "load_1h": "carga_1h",
        "load_24h": "carga_24h",
        "load_grupo_24h": "carga_grupo_24h",
    }
    out = text
    for k, v in mapping.items():
        out = out.replace(k, v)
    return out

def run(feat: pd.DataFrame | None = None) -> dict[str, Any]:
    if feat is None:
        feat = pd.read_parquet(cfg.FEATURES_PATH)
        feat["aberto"] = pd.to_datetime(feat["aberto"])

    cfg.MODELS.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUTS.mkdir(parents=True, exist_ok=True)

    train_end = pd.Timestamp(cfg.RISK_TRAIN_END)
    test_start = pd.Timestamp(cfg.RISK_TEST_START)

    train = feat[feat["aberto"] <= train_end]
    test = feat[feat["aberto"] >= test_start]
    print(f"risk: treino={len(train)} teste={len(test)} (até {train_end.date()} / a partir de {test_start.date()})")

    X_train = train[RISK_FEATURE_COLS]
    y_train = train["kpi_violado_flag"].astype(int).values
    X_test = test[RISK_FEATURE_COLS]
    y_test = test["kpi_violado_flag"].astype(int).values

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    spw = (n_neg / n_pos) if n_pos else 1.0

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        random_state=cfg.RANDOM_STATE,
        n_jobs=4,
    )
    clf.fit(X_train, y_train)

    split_idx = int(len(train) * 0.8)
    X_cal = X_train.iloc[split_idx:]
    y_cal = y_train[split_idx:]
    X_fit = X_train.iloc[:split_idx]
    y_fit = y_train[:split_idx]
    clf_cal = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        random_state=cfg.RANDOM_STATE,
        n_jobs=4,
    )
    clf_cal.fit(X_fit, y_fit)
    calibrator = CalibratedClassifierCV(clf_cal, method="sigmoid", cv="prefit")
    calibrator.fit(X_cal, y_cal)

    raw_proba = clf.predict_proba(X_test)[:, 1]
    cal_proba = calibrator.predict_proba(X_test)[:, 1]

    grupo_rate = train.groupby("grupo_designado")["kpi_violado_flag"].mean().to_dict()
    base_rate = float(y_train.mean())
    heur = test["grupo_designado"].map(lambda g: grupo_rate.get(g, base_rate)).values.astype(float)

    lr = LogisticRegression(max_iter=500, class_weight="balanced", random_state=cfg.RANDOM_STATE)
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]

    def pack_metrics(name: str, scores: np.ndarray) -> dict:
        pred = (scores >= 0.5).astype(int)
        try:
            auc = float(roc_auc_score(y_test, scores))
        except ValueError:
            auc = float("nan")
        try:
            ap = float(average_precision_score(y_test, scores))
        except ValueError:
            ap = float("nan")
        return {
            "model": name,
            "roc_auc": round(auc, 4) if auc == auc else None,
            "pr_auc": round(ap, 4) if ap == ap else None,
            "f1_0.5": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "brier": round(float(brier_score_loss(y_test, scores)), 4),
            "precision@50": round(precision_at_k(y_test, scores, 50), 4),
            "precision@100": round(precision_at_k(y_test, scores, 100), 4),
            "precision@200": round(precision_at_k(y_test, scores, 200), 4),
            "recall@50": round(recall_at_k(y_test, scores, 50), 4),
            "recall@100": round(recall_at_k(y_test, scores, 100), 4),
            "recall@200": round(recall_at_k(y_test, scores, 200), 4),
            "lift@5%": round(lift_at_pct(y_test, scores, 0.05), 4),
        }

    metrics = {
        "split": {
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_pos_train": n_pos,
            "n_pos_test": int(y_test.sum()),
            "base_rate_test": round(float(y_test.mean()), 4),
        },
        "xgboost_raw": pack_metrics("xgboost_raw", raw_proba),
        "xgboost_calibrated": pack_metrics("xgboost_calibrated", cal_proba),
        "baseline_grupo": pack_metrics("baseline_grupo", heur),
        "baseline_logistic": pack_metrics("baseline_logistic", lr_proba),
        "scale_pos_weight": round(spw, 2),
    }

    print("risk: SHAP...")
    factors, global_imp = shap_top_factors(clf, X_test, RISK_FEATURE_COLS)
    factors = [friendly_factor_names(t) for t in factors]
    metrics["feature_importance"] = {
        friendly_factor_names(k): round(v, 6) for k, v in sorted(global_imp.items(), key=lambda x: -x[1])
    }

    scores = test[
        ["numero", "aberto", "prioridade", "grupo_designado", "produto", "categoria", "origem", "descricao", "template", "item_config", "kpi_violado_flag"]
    ].copy()
    scores["score"] = cal_proba
    scores["score_raw"] = raw_proba
    scores["severidade"] = severity_from_scores(cal_proba)
    scores["fatores"] = factors
    scores = scores.sort_values("score", ascending=False)
    scores.to_parquet(cfg.ALERT_SCORES_PATH, index=False)

    joblib.dump(
        {
            "model": clf,
            "calibrator": calibrator,
            "features": RISK_FEATURE_COLS,
            "metrics": metrics,
        },
        cfg.MODELS / "xgboost.pkl",
    )

    cfg.RISK_METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"risk: AUC={metrics['xgboost_calibrated']['roc_auc']} PR-AUC={metrics['xgboost_calibrated']['pr_auc']}")
    print(f"risk: → {cfg.ALERT_SCORES_PATH}")
    return metrics

if __name__ == "__main__":
    run()
