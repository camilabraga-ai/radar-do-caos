from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from radar import config as cfg

def band_for_value(value: float, bands: list[tuple]) -> int:
    """Retorna o % de atingimento para um valor dado as faixas (lo, hi, pct)."""
    for lo, hi, pct in bands:
        lo_ok = True if lo is None else value >= lo
        hi_ok = True if hi is None else value <= hi
        if lo_ok and hi_ok:
            return pct
    return 0

def kpi_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Incidentes que entram no KPI: P2/P3, sem Sem Intervenção, sem Incidente Pai."""
    return df[
        df["eh_kpi"]
        & ~df["sem_intervencao"]
        & ~df["eh_filho"]
        & df["prioridade_num"].isin([2, 3])
    ].copy()

def ytd_stats(df: pd.DataFrame, as_of: pd.Timestamp, year: int = 2025) -> dict[str, Any]:
    uni = kpi_universe(df)
    uni = uni[(uni["aberto"] >= pd.Timestamp(f"{year}-01-01")) & (uni["aberto"] <= as_of)]
    out: dict[str, Any] = {"as_of": str(as_of.date()), "year": year}
    for p in (2, 3):
        sub = uni[uni["prioridade_num"] == p]
        volume = int(len(sub))
        ola = int(sub["kpi_violado_flag"].sum())
        out[f"p{p}"] = {
            "volume": volume,
            "ola_quebrados": ola,
            "taxa_ola": round(ola / volume, 4) if volume else 0.0,
            "faixa_volume": band_for_value(volume, cfg.KPI_VOLUME_BANDS[p]),
            "faixa_ola": band_for_value(ola, cfg.KPI_OLA_BANDS[p]),
        }
    return out

def monte_carlo_projection(
    df: pd.DataFrame,
    forecast: pd.DataFrame,
    as_of: pd.Timestamp,
    year: int = 2025,
    n_sims: int = cfg.MONTE_CARLO_N,
) -> dict[str, Any]:
    """Projeta totais anuais e probabilidade de cada faixa."""
    ytd = ytd_stats(df, as_of, year)
    year_end = pd.Timestamp(f"{year}-12-31")
    remaining_days = max(0, (year_end - as_of).days)

    result = {"as_of": str(as_of.date()), "remaining_days": remaining_days, "ytd": ytd, "priorities": {}}
    rng = np.random.default_rng(cfg.RANDOM_STATE)

    for p, serie_name in [(2, "kpi_p2"), (3, "kpi_p3")]:
        realized_vol = ytd[f"p{p}"]["volume"]
        realized_ola = ytd[f"p{p}"]["ola_quebrados"]
        recent = kpi_universe(df)
        recent = recent[
            (recent["prioridade_num"] == p)
            & (recent["aberto"] > as_of - pd.Timedelta(days=30))
            & (recent["aberto"] <= as_of)
        ]
        rate = float(recent["kpi_violado_flag"].mean()) if len(recent) else ytd[f"p{p}"]["taxa_ola"]
        rate = float(np.clip(rate, 0.001, 0.2))

        if remaining_days == 0:
            vol_dist = np.full(n_sims, realized_vol)
            ola_dist = np.full(n_sims, realized_ola)
        else:
            fc = forecast[(forecast["serie"] == serie_name) & (forecast["is_forecast"])]
            if len(fc) == 0:
                hist = forecast[
                    (forecast["serie"] == serie_name) & (~forecast["is_forecast"])
                ].sort_values("ds")
                daily_mean = float(hist["y"].tail(14).mean()) if len(hist) else 50.0
            else:
                daily_mean = float(fc["yhat"].mean())
            uni = kpi_universe(df)
            uni_p = uni[uni["prioridade_num"] == p]
            last8w = uni_p[uni_p["aberto"] > as_of - pd.Timedelta(days=56)]
            dow_mean = last8w.groupby(last8w["aberto"].dt.dayofweek).size()
            n_weeks = max(1, len(last8w["aberto"].dt.isocalendar().week.unique()))
            dow_avg = (dow_mean / n_weeks).to_dict()
            overall = float(daily_mean)

            days = pd.date_range(as_of + pd.Timedelta(days=1), year_end, freq="D")
            expected = np.array([dow_avg.get(d.dayofweek, overall) for d in days], dtype=float)
            if expected.mean() > 0 and daily_mean > 0:
                expected = expected * (daily_mean / expected.mean())

            future_vol = rng.poisson(np.clip(expected, 0.1, None), size=(n_sims, len(expected))).sum(axis=1)
            vol_dist = realized_vol + future_vol
            future_ola = rng.binomial(future_vol.astype(int), rate)
            ola_dist = realized_ola + future_ola

        vol_probs = {}
        ola_probs = {}
        for _, _, pct in cfg.KPI_VOLUME_BANDS[p]:
            vol_probs[str(pct)] = 0.0
        for _, _, pct in cfg.KPI_OLA_BANDS[p]:
            ola_probs[str(pct)] = 0.0
        for v in vol_dist:
            b = str(band_for_value(float(v), cfg.KPI_VOLUME_BANDS[p]))
            vol_probs[b] = vol_probs.get(b, 0) + 1
        for o in ola_dist:
            b = str(band_for_value(float(o), cfg.KPI_OLA_BANDS[p]))
            ola_probs[b] = ola_probs.get(b, 0) + 1
        for k in vol_probs:
            vol_probs[k] = round(vol_probs[k] / n_sims, 4)
        for k in ola_probs:
            ola_probs[k] = round(ola_probs[k] / n_sims, 4)

        result["priorities"][f"p{p}"] = {
            "volume_expected": round(float(vol_dist.mean()), 1),
            "ola_expected": round(float(ola_dist.mean()), 1),
            "rate_used": round(rate, 4),
            "prob_faixa_volume": vol_probs,
            "prob_faixa_ola": ola_probs,
            "prob_volume_ge_100": round(
                sum(vol_probs.get(str(p_), 0) for p_ in (150, 125, 100)), 4
            ),
            "prob_ola_ge_100": round(
                sum(ola_probs.get(str(p_), 0) for p_ in (150, 125, 100)), 4
            ),
            "vol_p10": round(float(np.percentile(vol_dist, 10)), 1),
            "vol_p90": round(float(np.percentile(vol_dist, 90)), 1),
            "ola_p10": round(float(np.percentile(ola_dist, 10)), 1),
            "ola_p90": round(float(np.percentile(ola_dist, 90)), 1),
        }
    return result

def ola_daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    uni = kpi_universe(df)
    rows = []
    for p in (2, 3):
        sub = uni[uni["prioridade_num"] == p]
        daily = sub.set_index("aberto").resample("D").agg(
            volume=("numero", "size"),
            ola=("kpi_violado_flag", "sum"),
        )
        daily["taxa"] = daily["ola"] / daily["volume"].clip(lower=1)
        daily["prioridade_num"] = p
        daily = daily.reset_index().rename(columns={"aberto": "ds"})
        rows.append(daily)
    for g in df["grupo_designado"].value_counts().head(6).index:
        sub = uni[uni["grupo_designado"] == g]
        if len(sub) < 50:
            continue
        daily = sub.set_index("aberto").resample("D").agg(
            volume=("numero", "size"),
            ola=("kpi_violado_flag", "sum"),
        )
        daily["taxa"] = daily["ola"] / daily["volume"].clip(lower=1)
        daily["grupo"] = g
        daily["prioridade_num"] = None
        daily = daily.reset_index().rename(columns={"aberto": "ds"})
        rows.append(daily)
    return pd.concat(rows, ignore_index=True)

def kpi_monthly(df: pd.DataFrame, year: int = 2025) -> pd.DataFrame:
    uni = kpi_universe(df)
    uni = uni[(uni["aberto"].dt.year == year)]
    uni["mes"] = uni["aberto"].dt.to_period("M").astype(str)
    rows = []
    for p in (2, 3):
        sub = uni[uni["prioridade_num"] == p]
        g = sub.groupby("mes").agg(volume=("numero", "size"), ola=("kpi_violado_flag", "sum")).reset_index()
        g["prioridade_num"] = p
        g["taxa"] = g["ola"] / g["volume"]
        g = g.sort_values("mes")
        g["volume_ytd"] = g["volume"].cumsum()
        g["ola_ytd"] = g["ola"].cumsum()
        g["faixa_volume_ytd"] = g["volume_ytd"].map(lambda v: band_for_value(v, cfg.KPI_VOLUME_BANDS[p]))
        g["faixa_ola_ytd"] = g["ola_ytd"].map(lambda v: band_for_value(v, cfg.KPI_OLA_BANDS[p]))
        rows.append(g)
    return pd.concat(rows, ignore_index=True)

def backtest_asof(df: pd.DataFrame, forecast: pd.DataFrame) -> list[dict]:
    """Valida projeção em datas as-of conhecidas vs realizado 2025."""
    actual = ytd_stats(df, pd.Timestamp("2025-12-31"), 2025)
    results = []
    for as_of_str in ("2025-06-30", "2025-09-30", "2025-11-30"):
        as_of = pd.Timestamp(as_of_str)
        proj = monte_carlo_projection(df, forecast, as_of, year=2025)
        for p in (2, 3):
            key = f"p{p}"
            results.append(
                {
                    "as_of": as_of_str,
                    "prioridade": p,
                    "proj_volume": proj["priorities"][key]["volume_expected"],
                    "real_volume": actual[key]["volume"],
                    "proj_ola": proj["priorities"][key]["ola_expected"],
                    "real_ola": actual[key]["ola_quebrados"],
                    "proj_faixa_volume_mode": max(
                        proj["priorities"][key]["prob_faixa_volume"],
                        key=proj["priorities"][key]["prob_faixa_volume"].get,
                    ),
                    "real_faixa_volume": actual[key]["faixa_volume"],
                    "proj_faixa_ola_mode": max(
                        proj["priorities"][key]["prob_faixa_ola"],
                        key=proj["priorities"][key]["prob_faixa_ola"].get,
                    ),
                    "real_faixa_ola": actual[key]["faixa_ola"],
                }
            )
    return results

def run(
    df: pd.DataFrame | None = None,
    forecast: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if df is None:
        df = pd.read_parquet(cfg.CURATED_PATH)
        df["aberto"] = pd.to_datetime(df["aberto"])
    if forecast is None:
        forecast = pd.read_parquet(cfg.FORECAST_PATH)
        forecast["ds"] = pd.to_datetime(forecast["ds"])

    cfg.OUTPUTS.mkdir(parents=True, exist_ok=True)
    as_of = pd.Timestamp(cfg.REF_DATE)
    print(f"kpi: projeção as-of {as_of.date()}...")

    projection = monte_carlo_projection(df, forecast, as_of)
    projection["backtest_asof"] = backtest_asof(df, forecast)
    projection["bands_legend"] = {
        "ola": {str(p): cfg.KPI_OLA_BANDS[p] for p in (2, 3)},
        "volume": {str(p): cfg.KPI_VOLUME_BANDS[p] for p in (2, 3)},
    }

    def fix(obj):
        if isinstance(obj, dict):
            return {k: fix(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [fix(x) for x in obj]
        if isinstance(obj, tuple):
            return [fix(x) for x in obj]
        return obj

    cfg.KPI_PROJECTION_PATH.write_text(
        json.dumps(fix(projection), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    ola = ola_daily_trend(df)
    ola.to_parquet(cfg.OLA_DAILY_PATH, index=False)
    monthly = kpi_monthly(df)
    monthly.to_parquet(cfg.KPI_MONTHLY_PATH, index=False)

    print(f"kpi: YTD P2 OLA={projection['ytd']['p2']['ola_quebrados']} vol={projection['ytd']['p2']['volume']}")
    print(f"kpi: YTD P3 OLA={projection['ytd']['p3']['ola_quebrados']} vol={projection['ytd']['p3']['volume']}")
    print(f"kpi: → {cfg.KPI_PROJECTION_PATH}")
    return projection

if __name__ == "__main__":
    run()
