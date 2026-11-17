

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from radar import config as cfg

RISK_FEATURE_COLS = [
    "prioridade_num",
    "grupo_enc",
    "produto_enc",
    "categoria_enc",
    "origem_enc",
    "hora",
    "dow",
    "fim_de_semana",
    "horario_comercial",
    "mes",
    "eh_filho",
    "feriado_br",
    "regime_pos_set25",
    "template_freq_30d",
    "ic_count_7d",
    "ic_count_30d",
    "grupo_viol_rate",
    "ic_viol_rate",
    "load_1h",
    "load_24h",
    "load_grupo_24h",
]

def _encode(series: pd.Series) -> tuple[pd.Series, LabelEncoder]:
    le = LabelEncoder()
    enc = le.fit_transform(series.astype(str))
    return pd.Series(enc, index=series.index), le

def build_risk_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:

    kpi = df[df["eh_kpi"]].copy().sort_values("aberto").reset_index(drop=True)
    if kpi.empty:
        raise RuntimeError("Subset KPI vazio")

    encoders: dict = {}
    for col, key in [
        ("grupo_designado", "grupo"),
        ("produto", "produto"),
        ("categoria", "categoria"),
        ("origem", "origem"),
    ]:
        encoded, le = _encode(kpi[col])
        kpi[f"{key}_enc" if key != "grupo" else "grupo_enc"] = encoded
        if key == "grupo":
            kpi["grupo_enc"] = encoded
        elif key == "produto":
            kpi["produto_enc"] = encoded
        elif key == "categoria":
            kpi["categoria_enc"] = encoded
        else:
            kpi["origem_enc"] = encoded
        encoders[key] = le

    kpi["fim_de_semana"] = kpi["fim_de_semana"].astype(int)
    kpi["horario_comercial"] = kpi["horario_comercial"].astype(int)
    kpi["eh_filho"] = kpi["eh_filho"].astype(int)
    kpi["feriado_br"] = kpi["feriado_br"].astype(int)
    kpi["regime_pos_set25"] = kpi["regime_pos_set25"].astype(int)

    kpi["template_freq_30d"] = 0.0
    kpi["ic_count_7d"] = 0.0
    kpi["ic_count_30d"] = 0.0
    kpi["grupo_viol_rate"] = 0.0
    kpi["ic_viol_rate"] = 0.0

    tpl_times: dict[str, list[pd.Timestamp]] = {}
    ic_times: dict[str, list[pd.Timestamp]] = {}
    grupo_hist: dict[str, list[tuple[pd.Timestamp, int]]] = {}
    ic_viol_hist: dict[str, list[tuple[pd.Timestamp, int]]] = {}

    t_freq = np.zeros(len(kpi))
    ic7 = np.zeros(len(kpi))
    ic30 = np.zeros(len(kpi))
    g_rate = np.zeros(len(kpi))
    ic_rate = np.zeros(len(kpi))

    for i, row in enumerate(kpi.itertuples(index=False)):
        t = row.aberto
        tpl = row.template_id
        ic = row.item_config
        g = row.grupo_designado
        viol = int(row.kpi_violado_flag)

        hist = tpl_times.setdefault(tpl, [])
        cutoff30 = t - pd.Timedelta(days=30)
        while hist and hist[0] < cutoff30:
            hist.pop(0)
        t_freq[i] = len(hist)
        hist.append(t)

        ih = ic_times.setdefault(ic, [])
        cutoff7 = t - pd.Timedelta(days=7)
        while ih and ih[0] < cutoff30:
            ih.pop(0)
        ic30[i] = len(ih)
        ic7[i] = sum(1 for x in ih if x >= cutoff7)
        ih.append(t)

        gh = grupo_hist.setdefault(g, [])
        if gh:
            n = len(gh)
            g_rate[i] = sum(v for _, v in gh) / n
        else:
            g_rate[i] = 0.0
        gh.append((t, viol))

        iv = ic_viol_hist.setdefault(ic, [])
        if iv:
            ic_rate[i] = sum(v for _, v in iv) / len(iv)
        else:
            ic_rate[i] = 0.0
        iv.append((t, viol))

    kpi["template_freq_30d"] = t_freq
    kpi["ic_count_7d"] = ic7
    kpi["ic_count_30d"] = ic30
    kpi["grupo_viol_rate"] = g_rate
    kpi["ic_viol_rate"] = ic_rate

    times = kpi["aberto"].values
    load_1h = np.zeros(len(kpi))
    load_24h = np.zeros(len(kpi))
    load_g24 = np.zeros(len(kpi))
    grupos = kpi["grupo_designado"].values
    j1 = 0
    j24 = 0
    for i in range(len(kpi)):
        t = times[i]
        while j1 < i and times[j1] < t - np.timedelta64(1, "h"):
            j1 += 1
        while j24 < i and times[j24] < t - np.timedelta64(24, "h"):
            j24 += 1
        load_1h[i] = i - j1
        load_24h[i] = i - j24
        g = grupos[i]
        load_g24[i] = sum(1 for k in range(j24, i) if grupos[k] == g)

    kpi["load_1h"] = load_1h
    kpi["load_24h"] = load_24h
    kpi["load_grupo_24h"] = load_g24

    for col in RISK_FEATURE_COLS:
        assert col not in cfg.LEAKAGE_COLS, f"leakage: {col}"
        assert col in kpi.columns, f"missing feature: {col}"

    keep = [
        "numero",
        "aberto",
        "prioridade",
        "prioridade_num",
        "grupo_designado",
        "produto",
        "categoria",
        "origem",
        "template",
        "template_id",
        "item_config",
        "descricao",
        "kpi_violado_flag",
        *RISK_FEATURE_COLS,
    ]
    keep = list(dict.fromkeys(keep))
    out = kpi[keep].copy()
    return out, encoders

def run(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_parquet(cfg.CURATED_PATH)
        df["aberto"] = pd.to_datetime(df["aberto"])
    print("features: construindo features de risco (sem leakage)...")
    feat, encoders = build_risk_features(df)
    cfg.OUTPUTS.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(cfg.FEATURES_PATH, index=False)
    print(f"features: {len(feat):,} linhas KPI → {cfg.FEATURES_PATH}".replace(",", "."))
    return feat

if __name__ == "__main__":
    run()
