from __future__ import annotations

import numpy as np
import pandas as pd

from radar import config as cfg

def robust_z(series: pd.Series, window: int = 28) -> pd.Series:
    med = series.rolling(window, min_periods=7).median()
    mad = (series - med).abs().rolling(window, min_periods=7).median()
    mad = mad.replace(0, np.nan)
    return 0.6745 * (series - med) / mad

def run(
    daily: pd.DataFrame | None = None,
    forecast: pd.DataFrame | None = None,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if daily is None:
        daily = pd.read_parquet(cfg.DAILY_SERIES_PATH)
        daily["ds"] = pd.to_datetime(daily["ds"])
    if forecast is None and cfg.FORECAST_PATH.exists():
        forecast = pd.read_parquet(cfg.FORECAST_PATH)
        forecast["ds"] = pd.to_datetime(forecast["ds"])
    if df is None:
        df = pd.read_parquet(cfg.CURATED_PATH)
        df["aberto"] = pd.to_datetime(df["aberto"])

    cfg.OUTPUTS.mkdir(parents=True, exist_ok=True)
    print("anomalies: z-score robusto por série/grupo/template...")

    rows = []
    for serie in ["total", "operacional", "kpi", "p2", "p3"]:
        s = daily[daily["serie"] == serie].sort_values("ds").set_index("ds")["y"]
        if len(s) < 30:
            continue
        z = robust_z(s)
        for ds, val in z.dropna().items():
            if abs(val) < 2.5:
                continue
            upper95 = None
            if forecast is not None:
                frow = forecast[(forecast["serie"] == serie) & (forecast["ds"] == ds) & (~forecast["is_forecast"])]
                if len(frow):
                    upper95 = float(frow.iloc[0]["yhat_upper"])
            severity = "Crítico" if abs(val) >= 4 else ("Alto" if abs(val) >= 3 else "Médio")
            rows.append(
                {
                    "ds": ds,
                    "dimensao": "serie",
                    "chave": serie,
                    "y": float(s.loc[ds]),
                    "z_score": round(float(val), 2),
                    "upper95": upper95,
                    "severidade": severity,
                }
            )

    top_g = df["grupo_designado"].value_counts().head(6).index
    for g in top_g:
        s = (
            df[df["grupo_designado"] == g]
            .set_index("aberto")
            .resample("D")
            .size()
        )
        z = robust_z(s)
        for ds, val in z.dropna().items():
            if abs(val) < 2.5:
                continue
            severity = "Crítico" if abs(val) >= 4 else ("Alto" if abs(val) >= 3 else "Médio")
            rows.append(
                {
                    "ds": ds,
                    "dimensao": "grupo",
                    "chave": g,
                    "y": float(s.loc[ds]),
                    "z_score": round(float(val), 2),
                    "upper95": None,
                    "severidade": severity,
                }
            )

    top_t = df["template"].value_counts().head(30).index
    for tpl in top_t:
        s = df[df["template"] == tpl].set_index("aberto").resample("D").size()
        if len(s) < 30:
            continue
        z = robust_z(s)
        for ds, val in z.dropna().items():
            if abs(val) < 3:
                continue
            severity = "Crítico" if abs(val) >= 4 else "Alto"
            rows.append(
                {
                    "ds": ds,
                    "dimensao": "template",
                    "chave": tpl[:80],
                    "y": float(s.loc[ds]),
                    "z_score": round(float(val), 2),
                    "upper95": None,
                    "severidade": severity,
                }
            )

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["ds", "z_score"], ascending=[False, False])
    out.to_parquet(cfg.ANOMALIES_PATH, index=False)
    print(f"anomalies: {len(out)} eventos → {cfg.ANOMALIES_PATH}")
    return out

if __name__ == "__main__":
    run()
