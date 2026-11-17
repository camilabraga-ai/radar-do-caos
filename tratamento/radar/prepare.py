from __future__ import annotations

import hashlib
import re
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from radar import config as cfg

def normalize_template(text: str) -> str:
    s = str(text) if text is not None else ""
    s = re.sub(r"IC\d+", "<IC>", s, flags=re.IGNORECASE)
    s = re.sub(r"\d+(?:\.\d+)?", "<N>", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s[:200]

def template_id(template: str) -> str:
    return hashlib.md5(template.encode("utf-8")).hexdigest()[:12]

def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = path or cfg.XLSX_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")
    df = pd.read_excel(path, sheet_name="Dataset Geral")
    return df.rename(columns=cfg.COL_MAP)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["aberto"] = pd.to_datetime(out["aberto"])
    out["resolvido"] = pd.to_datetime(out["resolvido"], errors="coerce")
    out["encerrado"] = pd.to_datetime(out["encerrado"], errors="coerce")

    out["prioridade_num"] = out["prioridade"].str.extract(r"^(\d)").astype(float)
    out["eh_kpi"] = out["entrou_kpi"].astype(str).str.upper() == "SIM"
    out["kpi_violado_flag"] = out["kpi_violado"].astype(str).str.upper() == "SIM"
    out["sla_limite_h"] = out["prioridade_num"].map(cfg.SLA_HOURS)
    out["sla_estourado"] = (out["eh_kpi"]) & (
        out["duracao"] > out["sla_limite_h"] * 3600
    )
    out["duracao_h"] = out["duracao"] / 3600.0
    out["eh_filho"] = out["incidente_pai"].notna() & (
        out["incidente_pai"].astype(str).str.strip() != ""
    )
    out["sem_intervencao"] = out["status"].astype(str) == "Sem Intervenção"
    out["origem"] = out["aberto_por"].astype(str)

    out["grupo_designado"] = out["grupo_designado"].fillna("Desconhecido")
    out["produto"] = out["produto"].fillna("NA")
    out["categoria"] = out["categoria"].fillna("NA")
    out["subcategoria"] = out["subcategoria"].fillna("NA")
    out["item_config"] = out["item_config"].fillna("NA")
    out["descricao"] = out["descricao"].fillna("")

    out["template"] = out["descricao"].map(normalize_template)
    out["template_id"] = out["template"].map(template_id)

    out["data"] = out["aberto"].dt.normalize()
    out["hora"] = out["aberto"].dt.hour
    out["dow"] = out["aberto"].dt.dayofweek  # 0=Mon
    out["dow_name"] = out["aberto"].dt.day_name()
    out["fim_de_semana"] = out["dow"] >= 5
    lo, hi = cfg.COMMERCIAL_HOURS
    out["horario_comercial"] = (out["hora"] >= lo) & (out["hora"] < hi)
    out["mes"] = out["aberto"].dt.month
    out["ano"] = out["aberto"].dt.year
    out["regime_pos_set25"] = out["aberto"] >= pd.Timestamp(cfg.REGIME_BREAK)

    br_holidays = holidays.Brazil(years=range(2023, 2027))
    out["feriado_br"] = out["data"].map(lambda d: d.date() in br_holidays)

    p99 = out["duracao_h"].quantile(0.99)
    out["duracao_h_cap"] = out["duracao_h"].clip(upper=p99)

    return out

def build_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """Séries diárias em formato longo: serie, ds, y."""
    rows: list[pd.DataFrame] = []

    def add(name: str, mask: pd.Series | None = None) -> None:
        sub = df if mask is None else df[mask]
        daily = sub.set_index("aberto").resample("D").size().rename("y").reset_index()
        daily.columns = ["ds", "y"]
        daily["serie"] = name
        rows.append(daily)

    add("total")
    add("operacional", ~df["sem_intervencao"])
    add("kpi", df["eh_kpi"])
    add("p2", df["prioridade_num"] == 2)
    add("p3", df["prioridade_num"] == 3)
    add("p4", df["prioridade_num"] == 4)
    add("kpi_p2", df["eh_kpi"] & (df["prioridade_num"] == 2))
    add("kpi_p3", df["eh_kpi"] & (df["prioridade_num"] == 3))
    add("viol_p2", df["kpi_violado_flag"] & (df["prioridade_num"] == 2))
    add("viol_p3", df["kpi_violado_flag"] & (df["prioridade_num"] == 3))

    top_g = df["grupo_designado"].value_counts().head(cfg.TOP_GRUPOS).index
    for g in top_g:
        add(f"grupo:{g}", df["grupo_designado"] == g)

    top_p = (
        df.loc[df["produto"] != "NA", "produto"].value_counts().head(cfg.TOP_PRODUTOS).index
    )
    for p in top_p:
        add(f"produto:{p}", df["produto"] == p)

    top_c = (
        df.loc[df["categoria"] != "NA", "categoria"]
        .value_counts()
        .head(cfg.TOP_CATEGORIAS)
        .index
    )
    for c in top_c:
        add(f"categoria:{c}", df["categoria"] == c)

    for origem in ("Manual", "Monitoramento"):
        add(f"origem:{origem}", df["origem"] == origem)

    out = pd.concat(rows, ignore_index=True)
    out["ds"] = pd.to_datetime(out["ds"])
    return out[["serie", "ds", "y"]]

def run(xlsx: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("prepare: carregando dataset...")
    raw = load_raw(xlsx)
    print(f"prepare: {len(raw):,} linhas brutas".replace(",", "."))
    df = clean(raw)
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.CURATED_PATH, index=False)
    print(f"prepare: curated → {cfg.CURATED_PATH}")

    series = build_daily_series(df)
    series.to_parquet(cfg.DAILY_SERIES_PATH, index=False)
    print(f"prepare: {series['serie'].nunique()} séries → {cfg.DAILY_SERIES_PATH}")
    return df, series

if __name__ == "__main__":
    run()
