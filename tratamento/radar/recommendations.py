from __future__ import annotations

import json

import numpy as np
import pandas as pd

from radar import config as cfg

def run(
    df: pd.DataFrame | None = None,
    forecast: pd.DataFrame | None = None,
    scores: pd.DataFrame | None = None,
    recurrence: pd.DataFrame | None = None,
    anomalies: pd.DataFrame | None = None,
    kpi_proj: dict | None = None,
    cluster_profiles: dict | None = None,
) -> pd.DataFrame:
    if df is None:
        df = pd.read_parquet(cfg.CURATED_PATH)
        df["aberto"] = pd.to_datetime(df["aberto"])
    if forecast is None:
        forecast = pd.read_parquet(cfg.FORECAST_PATH)
        forecast["ds"] = pd.to_datetime(forecast["ds"])
    if scores is None:
        scores = pd.read_parquet(cfg.ALERT_SCORES_PATH)
    if recurrence is None and cfg.RECURRENCE_PATH.exists():
        recurrence = pd.read_parquet(cfg.RECURRENCE_PATH)
    if anomalies is None and cfg.ANOMALIES_PATH.exists():
        anomalies = pd.read_parquet(cfg.ANOMALIES_PATH)
        anomalies["ds"] = pd.to_datetime(anomalies["ds"])
    if kpi_proj is None and cfg.KPI_PROJECTION_PATH.exists():
        kpi_proj = json.loads(cfg.KPI_PROJECTION_PATH.read_text(encoding="utf-8"))
    if cluster_profiles is None and cfg.CLUSTER_PROFILES_PATH.exists():
        cluster_profiles = json.loads(cfg.CLUSTER_PROFILES_PATH.read_text(encoding="utf-8"))

    cfg.OUTPUTS.mkdir(parents=True, exist_ok=True)
    recs: list[dict] = []
    rid = 0

    def add(**kwargs):
        nonlocal rid
        rid += 1
        recs.append({"id": f"REC-{rid:03d}", **kwargs})

    ref = pd.Timestamp(cfg.REF_DATE)

    for serie in ["total", "p2", "p3"] + [f"grupo:{g}" for g in df["grupo_designado"].value_counts().head(4).index]:
        hist = forecast[(forecast["serie"] == serie) & (~forecast["is_forecast"])].sort_values("ds")
        fut = forecast[(forecast["serie"] == serie) & (forecast["is_forecast"])].sort_values("ds")
        if hist.empty or fut.empty:
            continue
        p80 = float(hist["y"].tail(90).quantile(0.8))
        mean = float(hist["y"].tail(90).mean())
        peaks = fut[fut["yhat"] > p80]
        if peaks.empty:
            continue
        peak = peaks.loc[peaks["yhat"].idxmax()]
        pct = (float(peak["yhat"]) - mean) / max(mean, 1) * 100
        resp = serie.split(":")[-1] if serie.startswith("grupo:") else "Operação"
        add(
            tipo="capacidade",
            titulo=f"Pico previsto em {pd.Timestamp(peak['ds']).date()} ({serie})",
            evidencia=(
                f"Forecast {serie}: ~{peak['yhat']:.0f} incidentes "
                f"(p80 histórico 90d = {p80:.0f}, média {mean:.0f}, +{pct:.0f}%). "
                f"Modelo: {peak.get('modelo', 'n/d')}."
            ),
            acao=f"Reforçar plantão / capacidade de {resp} em {pd.Timestamp(peak['ds']).strftime('%d/%m')}.",
            responsavel=resp,
            prioridade="Alta" if pct > 30 else "Média",
            impacto_estimado=f"+{pct:.0f}% volume vs média",
            confianca=round(min(0.95, 0.55 + pct / 200), 2),
            fonte="Prophet/baseline",
            links=serie,
        )

    crit = scores[scores["severidade"].isin(["Crítico", "Alto"])]
    if len(crit):
        for g, grp in crit.groupby("grupo_designado"):
            if len(grp) < 2:
                continue
            top = grp.head(5)
            add(
                tipo="risco_kpi",
                titulo=f"Priorizar {len(grp)} chamados de risco no {g}",
                evidencia=(
                    f"{len(grp)} incidentes com severidade Crítico/Alto; "
                    f"probabilidade média {grp['score'].mean():.1%}. "
                    f"Fatores típicos: {top.iloc[0]['fatores']}. "
                    f"Exemplos: {', '.join(top['numero'].astype(str).head(3))}."
                ),
                acao=f"Fila prioritária no {g}: atacar top scores antes do estouro de OLA.",
                responsavel=str(g),
                prioridade="Crítica" if (grp["severidade"] == "Crítico").any() else "Alta",
                impacto_estimado=f"Até {len(grp)} violações evitáveis",
                confianca=round(float(grp["score"].mean()), 2),
                fonte="XGBoost+SHAP",
                links=",".join(top["numero"].astype(str).head(5)),
            )

    last90 = df[df["aberto"] > ref - pd.Timedelta(days=90)]
    tpl_stats = (
        last90.groupby("template")
        .agg(n=("numero", "size"), pct_si=("sem_intervencao", "mean"))
        .reset_index()
    )
    noise = tpl_stats[(tpl_stats["n"] >= 500) & (tpl_stats["pct_si"] >= 0.9)].sort_values("n", ascending=False)
    for _, row in noise.head(8).iterrows():
        monthly = row["n"] / 3
        add(
            tipo="ruido_monitoramento",
            titulo=f"Suprimir/ajustar alerta: {row['template'][:60]}",
            evidencia=(
                f"{int(row['n'])} ocorrências em 90d; "
                f"{row['pct_si']*100:.0f}% Sem Intervenção."
            ),
            acao="Revisar threshold, correlacionar ou agrupar o alerta no monitoramento.",
            responsavel="Monitoramento / NOC",
            prioridade="Média",
            impacto_estimado=f"~{monthly:.0f} chamados/mês evitáveis",
            confianca=0.85,
            fonte="regras+templates",
            links=row["template"][:80],
        )

    if "codigo_fechamento" in df.columns:
        closed = df[df["codigo_fechamento"].notna()]
        fp = (
            closed.groupby("template")["codigo_fechamento"]
            .apply(lambda s: s.isin(["Falso Positivo", "Falha não reproduzida"]).mean())
            .reset_index(name="pct_fp")
        )
        fp["n"] = closed.groupby("template").size().values
        bad = fp[(fp["pct_fp"] >= 0.15) & (fp["n"] >= 50)].sort_values("pct_fp", ascending=False)
        for _, row in bad.head(5).iterrows():
            add(
                tipo="falso_positivo",
                titulo=f"Alerta com alta taxa de FP: {str(row['template'])[:50]}",
                evidencia=f"{row['pct_fp']*100:.0f}% falso positivo / não reproduzida em {int(row['n'])} casos.",
                acao="Ajustar regra de monitoramento ou enriquecer correlação.",
                responsavel="Monitoramento",
                prioridade="Média",
                impacto_estimado="Redução de ruído operacional",
                confianca=0.75,
                fonte="código_fechamento",
                links=str(row["template"])[:80],
            )

    if recurrence is not None and len(recurrence):
        hot = recurrence[recurrence["n_30d"] >= 10].head(15)
        for _, row in hot.iterrows():
            add(
                tipo="recorrencia",
                titulo=f"Abrir Problema: {row['item_config']} × template recorrente",
                evidencia=(
                    f"{int(row['n_30d'])} ocorrências em 30d "
                    f"(90d: {int(row['n_90d'])}); intervalo médio {row['intervalo_medio_d']}d; "
                    f"{row['pct_sem_intervencao']}% Sem Intervenção. "
                    f"Template: {str(row['template'])[:70]}"
                ),
                acao=f"Abrir registro de Problema para IC {row['item_config']} e eliminar causa raiz.",
                responsavel=str(row.get("grupo_designado") or "Operação"),
                prioridade="Alta" if row["n_30d"] >= 30 else "Média",
                impacto_estimado=f"-{int(row['n_30d'])} incidentes/mês potenciais",
                confianca=0.8,
                fonte="recorrência IC×template",
                links=f"{row['item_config']}",
            )

    if cfg.OLA_DAILY_PATH.exists():
        ola = pd.read_parquet(cfg.OLA_DAILY_PATH)
        ola["ds"] = pd.to_datetime(ola["ds"])
        for p in (2, 3):
            sub = ola[(ola["prioridade_num"] == p) & (ola["ds"] > ref - pd.Timedelta(days=42))]
            if len(sub) < 14:
                continue
            weekly = sub.set_index("ds").resample("W")["ola"].sum()
            if len(weekly) >= 4:
                slope = np.polyfit(range(len(weekly)), weekly.values.astype(float), 1)[0]
                if slope > 0.3:
                    add(
                        tipo="tendencia_ola",
                        titulo=f"Perda de OLA P{p} em alta nas últimas semanas",
                        evidencia=f"Inclinação semanal de violações ≈ +{slope:.2f}/semana (6 semanas).",
                        acao=f"Revisar capacidade e filas P{p}; cruzar com alertas XGBoost.",
                        responsavel="Gestão de Operações",
                        prioridade="Alta",
                        impacto_estimado="Proteção da faixa de atingimento do KPI OLA",
                        confianca=0.7,
                        fonte="OLA daily",
                        links=f"p{p}",
                    )

    if kpi_proj and "priorities" in kpi_proj and int(kpi_proj.get("remaining_days") or 0) > 0:
        for p in (2, 3):
            key = f"p{p}"
            info = kpi_proj["priorities"].get(key, {})
            for metric, prob_key in [("OLA", "prob_ola_ge_100"), ("volume", "prob_volume_ge_100")]:
                prob = info.get(prob_key, 1.0)
                if prob < 0.70:
                    add(
                        tipo="atingimento_kpi",
                        titulo=f"Risco de não atingir 100% no KPI {metric} P{p}",
                        evidencia=(
                            f"P(faixa ≥ 100%) = {prob:.0%}. "
                            f"Esperado: {info.get('ola_expected' if metric=='OLA' else 'volume_expected')}."
                        ),
                        acao="Plano de contenção: priorizar violações previsíveis e reduzir volume evitável (ruído).",
                        responsavel="Gestão de Operações",
                        prioridade="Crítica" if prob < 0.5 else "Alta",
                        impacto_estimado=f"Recuperar faixa de atingimento {metric} P{p}",
                        confianca=0.75,
                        fonte="Monte Carlo KPI",
                        links=key,
                    )
    elif kpi_proj and "ytd" in kpi_proj:
        for p in (2, 3):
            y = kpi_proj["ytd"].get(f"p{p}", {})
            if y.get("faixa_ola", 100) < 100:
                add(
                    tipo="atingimento_kpi",
                    titulo=f"KPI OLA P{p} fechou abaixo de 100% (faixa {y.get('faixa_ola')}%)",
                    evidencia=(
                        f"YTD: {y.get('ola_quebrados')} OLA quebrados em {y.get('volume')} incidentes KPI. "
                        f"Faixa oficial: {y.get('faixa_ola')}%."
                    ),
                    acao="Retrospectiva: causas das violações e plano 2026 (capacidade + redução de ruído).",
                    responsavel="Gestão de Operações",
                    prioridade="Alta",
                    impacto_estimado="Aprendizado para meta do próximo ciclo",
                    confianca=0.9,
                    fonte="KPI YTD",
                    links=f"p{p}",
                )

    if cluster_profiles and "clusters" in cluster_profiles:
        for cid, prof in cluster_profiles["clusters"].items():
            if prof.get("trend_30d", 0) > 0.5 and prof.get("n_30d", 0) >= 100:
                add(
                    tipo="padrao_crescente",
                    titulo=f"Cluster #{cid} em alta: {', '.join(prof.get('top_terms', [])[:3])}",
                    evidencia=(
                        f"Tendência 30d = {prof['trend_30d']:+.0%}; "
                        f"{prof['n_30d']} vs {prof['n_prev_30d']} no período anterior; "
                        f"{prof['pct_sem_intervencao']}% Sem Intervenção."
                    ),
                    acao="Investigar causa do padrão e avaliar automação/supressão.",
                    responsavel=str(prof.get("top_grupo") or "Operação"),
                    prioridade="Média",
                    impacto_estimado=f"Cluster tamanho {prof['size']}",
                    confianca=0.65,
                    fonte="KMeans TF-IDF",
                    links=f"cluster:{cid}",
                )

    out = pd.DataFrame(recs)
    if len(out):
        prio_order = {"Crítica": 0, "Alta": 1, "Média": 2, "Baixa": 3}
        out["_o"] = out["prioridade"].map(prio_order).fillna(9)
        out = out.sort_values(["_o", "confianca"], ascending=[True, False]).drop(columns=["_o"])
    out.to_parquet(cfg.RECOMMENDATIONS_PATH, index=False)
    print(f"recommendations: {len(out)} cards → {cfg.RECOMMENDATIONS_PATH}")
    return out

if __name__ == "__main__":
    run()
