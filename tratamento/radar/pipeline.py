

from __future__ import annotations

import argparse
import time

import pandas as pd

from radar import config as cfg
from radar import anomalies, features, forecast, kpi, patterns, prepare, recommendations, registry, risk

STAGES = [
    "prepare",
    "features",
    "forecast",
    "risk",
    "patterns",
    "anomalies",
    "kpi",
    "recommendations",
    "registry",
]

def run_stage(name: str, ctx: dict) -> dict:
    t0 = time.time()
    print(f"\n=== STAGE: {name} ===")
    if name == "prepare":
        df, series = prepare.run()
        ctx["df"] = df
        ctx["daily"] = series
        result = {"n_rows": len(df), "n_series": int(series["serie"].nunique())}
    elif name == "features":
        df = ctx.get("df")
        if df is None:
            df = pd.read_parquet(cfg.CURATED_PATH)
            df["aberto"] = pd.to_datetime(df["aberto"])
            ctx["df"] = df
        feat = features.run(df)
        ctx["feat"] = feat
        result = {"n_kpi": len(feat)}
    elif name == "forecast":
        daily = ctx.get("daily")
        if daily is None:
            daily = pd.read_parquet(cfg.DAILY_SERIES_PATH)
            daily["ds"] = pd.to_datetime(daily["ds"])
            ctx["daily"] = daily
        result = forecast.run(daily)
    elif name == "risk":
        feat = ctx.get("feat")
        if feat is None:
            feat = pd.read_parquet(cfg.FEATURES_PATH)
            feat["aberto"] = pd.to_datetime(feat["aberto"])
        result = risk.run(feat)
    elif name == "patterns":
        df = ctx.get("df")
        if df is None:
            df = pd.read_parquet(cfg.CURATED_PATH)
            df["aberto"] = pd.to_datetime(df["aberto"])
            ctx["df"] = df
        result = patterns.run(df)
    elif name == "anomalies":
        result = {"n": len(anomalies.run(ctx.get("daily"), None, ctx.get("df")))}
    elif name == "kpi":
        df = ctx.get("df")
        if df is None:
            df = pd.read_parquet(cfg.CURATED_PATH)
            df["aberto"] = pd.to_datetime(df["aberto"])
        fc = pd.read_parquet(cfg.FORECAST_PATH)
        fc["ds"] = pd.to_datetime(fc["ds"])
        result = kpi.run(df, fc)
    elif name == "recommendations":
        result = {"n": len(recommendations.run())}
    elif name == "registry":
        result = registry.update(ctx.get("stage_results", {}))
    else:
        raise ValueError(f"Stage desconhecido: {name}")
    elapsed = round(time.time() - t0, 1)
    print(f"=== {name} OK ({elapsed}s) ===")
    return {"elapsed_s": elapsed, "result_keys": list(result.keys()) if isinstance(result, dict) else type(result).__name__}

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Radar do Caos — pipeline")
    parser.add_argument(
        "--stage",
        default="all",
        help=f"Estágio a rodar: all | {' | '.join(STAGES)}",
    )
    args = parser.parse_args(argv)
    stages = STAGES if args.stage == "all" else [args.stage]
    ctx: dict = {"stage_results": {}}
    for s in stages:
        if s == "registry":
            continue
        meta = run_stage(s, ctx)
        ctx["stage_results"][s] = meta
    if args.stage in ("all", "registry"):
        run_stage("registry", ctx)
    print("\nPipeline concluído.")

if __name__ == "__main__":
    main()
