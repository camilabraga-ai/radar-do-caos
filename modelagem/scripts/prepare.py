#!/usr/bin/env python3

from __future__ import annotations

import re
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_XLSX = ROOT / "data" / "raw" / "LW-DATASET.xlsx"
CURATED = ROOT / "data" / "curated.parquet"
DAILY = ROOT / "outputs" / "daily.parquet"
FEATURES = ROOT / "outputs" / "daily_features.parquet"

SLA_HOURS = {1: 4, 2: 4, 3: 12, 4: 24, 5: 96}
REGIME_START = pd.Timestamp("2025-09-01")

COL_MAP = {
    "Número": "numero",
    "Prioridade": "prioridade",
    "Produto": "produto",
    "Categoria": "categoria",
    "Subcategoria": "subcategoria",
    "Grupo designado": "grupo_designado",
    "Item de configuração": "item_config",
    "Aberto": "aberto",
    "Resolvido": "resolvido",
    "Encerrado": "encerrado",
    "Duração": "duracao",
    "Descrição resumida": "descricao",
    "Aberto por": "aberto_por",
    "Incidente Pai": "incidente_pai",
    "Status": "status",
    "Entrou para KPI?": "entrou_kpi",
    "KPI Violado?": "kpi_violado",
}

KEEP = [
    "numero",
    "aberto",
    "data",
    "ano",
    "mes",
    "dow",
    "hora",
    "prioridade",
    "prioridade_num",
    "produto",
    "categoria",
    "item_config",
    "grupo_designado",
    "descricao",
    "template",
    "origem",
    "status",
    "eh_filho",
    "eh_kpi",
    "kpi_violado_flag",
    "sla_estourado",
]

def normalize_template(text: object) -> str:
    s = "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text)
    s = re.sub(r"IC\d+", "<IC>", s, flags=re.IGNORECASE)
    s = re.sub(r"\d+(?:\.\d+)?", "<N>", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s[:200]

def load_and_clean() -> pd.DataFrame:
    if not DATA_XLSX.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {DATA_XLSX}")
    df = pd.read_excel(DATA_XLSX, sheet_name="Dataset Geral")
    df = df.rename(columns=COL_MAP)
    df["aberto"] = pd.to_datetime(df["aberto"])
    df["prioridade_num"] = df["prioridade"].astype(str).str.extract(r"^(\d)").astype(float)
    df["eh_kpi"] = df["entrou_kpi"].astype(str).str.upper() == "SIM"
    df["kpi_violado_flag"] = df["kpi_violado"].astype(str).str.upper() == "SIM"
    df["sla_limite_horas"] = df["prioridade_num"].map(SLA_HOURS)
    duracao = pd.to_numeric(df["duracao"], errors="coerce")
    df["sla_estourado"] = df["eh_kpi"] & (duracao > df["sla_limite_horas"] * 3600)
    df["data"] = df["aberto"].dt.normalize()
    df["ano"] = df["aberto"].dt.year
    df["mes"] = df["aberto"].dt.month
    df["dow"] = df["aberto"].dt.dayofweek
    df["hora"] = df["aberto"].dt.hour
    df["grupo_designado"] = df["grupo_designado"].fillna("Desconhecido")
    df["produto"] = df["produto"].replace("", pd.NA)
    df["categoria"] = df["categoria"].replace("", pd.NA)
    df["item_config"] = df["item_config"].replace("", pd.NA)
    df["origem"] = df["aberto_por"].fillna("Desconhecido")
    df["eh_filho"] = df["incidente_pai"].notna() & (df["incidente_pai"].astype(str).str.strip() != "")
    df["template"] = df["descricao"].map(normalize_template)
    return df[KEEP].copy()

def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    def _count(mask: pd.Series, serie: str) -> None:
        g = df.loc[mask].groupby("data").size().rename("y").reset_index()
        g["serie"] = serie
        frames.append(g.rename(columns={"data": "ds"}))

    _count(pd.Series(True, index=df.index), "total")
    _count(df["prioridade_num"] == 2, "p2")
    _count(df["prioridade_num"] == 3, "p3")
    _count(df["eh_kpi"] & (df["prioridade_num"] == 2) & df["kpi_violado_flag"], "ola_p2")
    _count(df["eh_kpi"] & (df["prioridade_num"] == 3) & df["kpi_violado_flag"], "ola_p3")

    raw = pd.concat(frames, ignore_index=True)
    start, end = raw["ds"].min(), raw["ds"].max()
    idx = pd.date_range(start, end, freq="D")
    parts = []
    for serie, g in raw.groupby("serie"):
        s = g.set_index("ds")["y"].reindex(idx, fill_value=0).rename("y").reset_index()
        s.columns = ["ds", "y"]
        s["serie"] = serie
        parts.append(s)
    return pd.concat(parts, ignore_index=True)

def build_daily_context(df: pd.DataFrame) -> pd.DataFrame:

    g = df.groupby("data").agg(
        n=("numero", "size"),
        n_mon=("origem", lambda s: (s.astype(str).str.contains("Monitor", case=False, na=False)).sum()),
        n_kpi=("eh_kpi", "sum"),
        n_p2=("prioridade_num", lambda s: (s == 2).sum()),
        n_p3=("prioridade_num", lambda s: (s == 3).sum()),
        n_sem_int=("status", lambda s: (s.astype(str).str.contains("Sem Interven", case=False, na=False)).sum()),
    )
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D")).fillna(0)
    ctx = pd.DataFrame({"ds": g.index})
    n = g["n"].replace(0, np.nan)
    ctx["pct_monitoramento"] = (g["n_mon"] / n).fillna(0).values
    ctx["pct_kpi"] = (g["n_kpi"] / n).fillna(0).values
    ctx["pct_p2"] = (g["n_p2"] / n).fillna(0).values
    ctx["pct_p3"] = (g["n_p3"] / n).fillna(0).values
    ctx["pct_sem_intervencao"] = (g["n_sem_int"] / n).fillna(0).values
    for c in ("pct_monitoramento", "pct_kpi", "pct_p2", "pct_p3", "pct_sem_intervencao"):
        ctx[c] = ctx[c].shift(1).bfill().fillna(0)
    return ctx

def build_features(daily: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    br = holidays.country_holidays("BR")
    parts = []
    for serie, g in daily.groupby("serie"):
        s = g.sort_values("ds").copy()
        s["ds"] = pd.to_datetime(s["ds"])
        s = s.merge(ctx, on="ds", how="left")

        s["regime_pos_set25"] = (s["ds"] >= REGIME_START).astype(int)
        s["feriado_br"] = s["ds"].dt.date.map(lambda d: int(d in br))
        s["feriado_vespera"] = (s["ds"] + pd.Timedelta(days=1)).dt.date.map(lambda d: int(d in br))
        s["dow"] = s["ds"].dt.dayofweek
        s["is_weekend"] = (s["dow"] >= 5).astype(int)
        s["month"] = s["ds"].dt.month
        s["day"] = s["ds"].dt.day
        s["weekofyear"] = s["ds"].dt.isocalendar().week.astype(int)
        s["is_month_start"] = (s["day"] <= 3).astype(int)
        s["is_month_end"] = (s["day"] >= 28).astype(int)

        for k in (1, 2, 3, 7, 14, 21, 28):
            s[f"y_lag_{k}"] = s["y"].shift(k)

        past = s["y"].shift(1)
        s["y_roll3_mean"] = past.rolling(3, min_periods=1).mean()
        s["y_roll7_mean"] = past.rolling(7, min_periods=1).mean()
        s["y_roll7_std"] = past.rolling(7, min_periods=1).std().fillna(0)
        s["y_roll14_mean"] = past.rolling(14, min_periods=1).mean()
        s["y_roll28_mean"] = past.rolling(28, min_periods=1).mean()
        s["y_ema7"] = past.ewm(span=7, adjust=False).mean()
        s["y_ema28"] = past.ewm(span=28, adjust=False).mean()

        s["y_diff_1"] = s["y_lag_1"] - s["y_lag_2"]
        s["y_diff_7"] = s["y_lag_1"] - s["y_lag_7"]
        s["y_ratio_7"] = s["y_lag_1"] / s["y_lag_7"].replace(0, np.nan)
        s["y_ratio_7"] = s["y_ratio_7"].fillna(1.0).clip(0.1, 10)
        s["y_vs_roll7"] = s["y_lag_1"] - s["y_roll7_mean"]
        s["y_vs_roll28"] = s["y_lag_1"] - s["y_roll28_mean"]

        s["y_dow_mean4"] = (
            s["y"].shift(7) + s["y"].shift(14) + s["y"].shift(21) + s["y"].shift(28)
        ) / 4.0

        s["regime_x_dow"] = s["regime_pos_set25"] * s["dow"]
        s["regime_x_mon"] = s["regime_pos_set25"] * s["pct_monitoramento"]

        parts.append(s)
    return pd.concat(parts, ignore_index=True)

def main() -> None:
    print("Carregando e limpando dataset...")
    df = load_and_clean()
    CURATED.parent.mkdir(parents=True, exist_ok=True)
    DAILY.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CURATED, index=False)
    daily = build_daily(df)
    daily.to_parquet(DAILY, index=False)
    ctx = build_daily_context(df)
    feats = build_features(daily, ctx)
    feats.to_parquet(FEATURES, index=False)
    print(f"OK: {len(df):,} linhas → {CURATED}".replace(",", "."))
    print(f"OK: {len(daily):,} pontos diários → {DAILY}".replace(",", "."))
    print(f"OK: {len(feats):,} features → {FEATURES}".replace(",", "."))
    print(f"   colunas FE: {len(feats.columns)}")

if __name__ == "__main__":
    main()
