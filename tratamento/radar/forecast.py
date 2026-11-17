from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet

from radar import config as cfg

warnings.filterwarnings("ignore")

CORE_SERIES = [
    "total",
    "operacional",
    "kpi",
    "p2",
    "p3",
    "p4",
    "kpi_p2",
    "kpi_p3",
    "viol_p2",
    "viol_p3",
    "origem:Manual",
    "origem:Monitoramento",
]

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

def baseline_seasonal(train: pd.DataFrame, horizon: int) -> np.ndarray:
    """Naive sazonal: mesmo DOW da semana anterior."""
    y = train.set_index("ds")["y"]
    preds = []
    last = y.index.max()
    for h in range(1, horizon + 1):
        target = last + pd.Timedelta(days=h)
        ref = target - pd.Timedelta(days=7)
        if ref in y.index:
            preds.append(float(y.loc[ref]))
        else:
            preds.append(float(y.tail(7).mean()))
    return np.array(preds)

def baseline_ma7(train: pd.DataFrame, horizon: int) -> np.ndarray:
    mean = float(train["y"].tail(7).mean())
    return np.full(horizon, mean)

def fit_prophet(train: pd.DataFrame, interval_width: float = 0.95) -> Prophet:
    n_days = (train["ds"].max() - train["ds"].min()).days
    yearly = n_days >= 365
    regime = pd.Timestamp(cfg.REGIME_BREAK)
    post = train[train["ds"] >= regime]
    use = post if len(post) >= 60 else train

    m = Prophet(
        interval_width=interval_width,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=yearly,
        changepoint_prior_scale=0.1,
    )
    try:
        m.add_country_holidays(country_name="BR")
    except Exception:
        pass
    if use["ds"].min() < regime <= use["ds"].max():
        m.changepoints = pd.Series([regime])
    m.fit(use[["ds", "y"]])
    return m

def predict_prophet(model: Prophet, last_ds: pd.Timestamp, horizon: int) -> pd.DataFrame:
    future = pd.DataFrame({"ds": pd.date_range(last_ds + pd.Timedelta(days=1), periods=horizon, freq="D")})
    fc = model.predict(future)
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]

def evaluate_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "smape": smape(y_true, y_pred),
    }

def backtest_series(daily: pd.DataFrame, serie: str) -> list[dict[str, Any]]:
    """Rolling origin weekly cuts from FORECAST_BACKTEST_START to END."""
    s = daily[daily["serie"] == serie][["ds", "y"]].sort_values("ds").reset_index(drop=True)
    if len(s) < 90:
        return []

    cuts = pd.date_range(cfg.FORECAST_BACKTEST_START, cfg.FORECAST_BACKTEST_END, freq="W-SUN")
    results = []
    for cut in cuts:
        train = s[s["ds"] <= cut]
        test = s[(s["ds"] > cut) & (s["ds"] <= cut + pd.Timedelta(days=cfg.HORIZON_DAYS))]
        if len(train) < 60 or len(test) < 1:
            continue
        h = len(test)
        y_true = test["y"].values.astype(float)

        b_seas = baseline_seasonal(train, h)
        b_ma = baseline_ma7(train, h)

        try:
            model = fit_prophet(train, interval_width=0.8)
            last = train["ds"].max()
            fc = predict_prophet(model, last, h)
            y_prophet = fc["yhat"].values[:h]
            lower = fc["yhat_lower"].values[:h]
            upper = fc["yhat_upper"].values[:h]
            coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
        except Exception as e:
            print(f"  prophet fail {serie} @ {cut.date()}: {e}")
            y_prophet = b_ma.copy()
            coverage = float("nan")

        for name, pred in [("prophet", y_prophet), ("seasonal", b_seas), ("ma7", b_ma)]:
            m1 = evaluate_horizon(y_true[:1], pred[:1]) if h >= 1 else {}
            mall = evaluate_horizon(y_true, pred)
            results.append(
                {
                    "serie": serie,
                    "cut": str(cut.date()),
                    "model": name,
                    "horizon_n": h,
                    "mae_d1": m1.get("mae"),
                    "smape_d1": m1.get("smape"),
                    "mae_d7": mall["mae"],
                    "rmse_d7": mall["rmse"],
                    "smape_d7": mall["smape"],
                    "coverage_80": coverage if name == "prophet" else None,
                }
            )
    return results

SEASONAL_TIE_PP = 5.0

def pick_champion(backtest_df: pd.DataFrame, serie: str) -> str:
    sub = backtest_df[backtest_df["serie"] == serie]
    if sub.empty:
        return "seasonal"
    agg = sub.groupby("model")["smape_d7"].mean()
    best = str(agg.idxmin())
    if best == "ma7" and "seasonal" in agg.index:
        if float(agg["seasonal"] - agg["ma7"]) <= SEASONAL_TIE_PP:
            return "seasonal"
    return best

def final_forecast(daily: pd.DataFrame, serie: str, champion: str) -> pd.DataFrame:
    s = daily[daily["serie"] == serie][["ds", "y"]].sort_values("ds")
    ref = pd.Timestamp(cfg.REF_DATE)
    train = s[s["ds"] <= ref]
    last = train["ds"].max()
    h = cfg.HORIZON_DAYS

    hist = train.copy()
    hist["yhat"] = hist["y"]
    hist["yhat_lower"] = hist["y"]
    hist["yhat_upper"] = hist["y"]
    hist["yhat_lower80"] = hist["y"]
    hist["yhat_upper80"] = hist["y"]
    hist["is_forecast"] = False
    hist["modelo"] = champion

    if champion == "prophet":
        try:
            m95 = fit_prophet(train, interval_width=0.95)
            m80 = fit_prophet(train, interval_width=0.8)
            fc95 = predict_prophet(m95, last, h)
            fc80 = predict_prophet(m80, last, h)
            fut = fc95.copy()
            fut["yhat_lower80"] = fc80["yhat_lower"].values
            fut["yhat_upper80"] = fc80["yhat_upper"].values
            fut = fut.rename(columns={"yhat_lower": "yhat_lower", "yhat_upper": "yhat_upper"})
            fut["y"] = np.nan
            fut["is_forecast"] = True
            fut["modelo"] = "prophet"
            fut = fut.rename(columns={"yhat_lower": "yhat_lower", "yhat_upper": "yhat_upper"})
            fut = fut.assign(
                yhat_lower=fut["yhat_lower"],
                yhat_upper=fut["yhat_upper"],
            )
        except Exception as e:
            print(f"  final prophet fail {serie}: {e}, fallback ma7")
            champion = "ma7"

    if champion == "seasonal":
        preds = baseline_seasonal(train, h)
        ds = pd.date_range(last + pd.Timedelta(days=1), periods=h, freq="D")
        std = float(train["y"].tail(28).std() or 1)
        fut = pd.DataFrame(
            {
                "ds": ds,
                "y": np.nan,
                "yhat": preds,
                "yhat_lower": preds - 1.96 * std,
                "yhat_upper": preds + 1.96 * std,
                "yhat_lower80": preds - 1.28 * std,
                "yhat_upper80": preds + 1.28 * std,
                "is_forecast": True,
                "modelo": "seasonal",
            }
        )
    elif champion == "ma7":
        preds = baseline_ma7(train, h)
        ds = pd.date_range(last + pd.Timedelta(days=1), periods=h, freq="D")
        std = float(train["y"].tail(28).std() or 1)
        fut = pd.DataFrame(
            {
                "ds": ds,
                "y": np.nan,
                "yhat": preds,
                "yhat_lower": np.maximum(0, preds - 1.96 * std),
                "yhat_upper": preds + 1.96 * std,
                "yhat_lower80": np.maximum(0, preds - 1.28 * std),
                "yhat_upper80": preds + 1.28 * std,
                "is_forecast": True,
                "modelo": "ma7",
            }
        )

    hist_out = hist[["ds", "y", "yhat", "yhat_lower", "yhat_upper", "yhat_lower80", "yhat_upper80", "is_forecast", "modelo"]]
    for c in hist_out.columns:
        if c not in fut.columns:
            fut[c] = np.nan
    fut_out = fut[hist_out.columns]
    out = pd.concat([hist_out, fut_out], ignore_index=True)
    out["serie"] = serie
    return out

def series_list(daily: pd.DataFrame) -> list[str]:
    available = set(daily["serie"].unique())
    core = [s for s in CORE_SERIES if s in available]
    extras = sorted(
        s for s in available if s.startswith(("grupo:", "produto:", "categoria:"))
    )
    return core, extras

def run(daily: pd.DataFrame | None = None) -> dict:
    if daily is None:
        daily = pd.read_parquet(cfg.DAILY_SERIES_PATH)
        daily["ds"] = pd.to_datetime(daily["ds"])

    cfg.OUTPUTS.mkdir(parents=True, exist_ok=True)
    core, extras = series_list(daily)
    print(f"forecast: backtest em {len(core)} séries core; forecast final em {len(core)+len(extras)}...")

    all_bt: list[dict] = []
    for i, serie in enumerate(core):
        print(f"  [{i+1}/{len(core)}] backtest {serie}")
        all_bt.extend(backtest_series(daily, serie))

    bt = pd.DataFrame(all_bt)
    bt.to_parquet(cfg.FORECAST_BACKTEST_PATH, index=False)

    champions: dict[str, str] = {}
    metrics_summary: dict[str, Any] = {"series": {}}
    for serie in core:
        ch = pick_champion(bt, serie) if not bt.empty else "ma7"
        champions[serie] = ch
        sub = bt[bt["serie"] == serie]
        metrics_summary["series"][serie] = {"champion": ch, "by_model": {}}
        if not sub.empty:
            for model, g in sub.groupby("model"):
                metrics_summary["series"][serie]["by_model"][model] = {
                    "smape_d1_mean": round(float(g["smape_d1"].mean()), 2),
                    "smape_d7_mean": round(float(g["smape_d7"].mean()), 2),
                    "mae_d7_mean": round(float(g["mae_d7"].mean()), 2),
                    "n_folds": int(len(g)),
                }

    default_ch = champions.get("total", "seasonal")
    for serie in extras:
        champions[serie] = default_ch
        metrics_summary["series"][serie] = {
            "champion": default_ch,
            "by_model": {},
            "note": "campeão herdado de total (sem backtest dedicado)",
        }

    print("forecast: previsão final...")
    frames = []
    for serie in core + extras:
        frames.append(final_forecast(daily, serie, champions[serie]))
    fc = pd.concat(frames, ignore_index=True)
    fc.to_parquet(cfg.FORECAST_PATH, index=False)

    metrics_summary["champions"] = champions
    metrics_summary["ref_date"] = cfg.REF_DATE
    metrics_summary["horizon_days"] = cfg.HORIZON_DAYS
    metrics_summary["backtest_cuts"] = bt["cut"].nunique() if not bt.empty else 0
    cfg.FORECAST_METRICS_PATH.write_text(
        json.dumps(metrics_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"forecast: → {cfg.FORECAST_PATH}")
    print(f"forecast: métricas → {cfg.FORECAST_METRICS_PATH}")
    return metrics_summary

if __name__ == "__main__":
    run()
