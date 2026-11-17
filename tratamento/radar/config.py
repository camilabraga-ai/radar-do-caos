

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_DIR = ROOT / "data"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"

XLSX_PATH = DATA_RAW / "LW-DATASET.xlsx"
CURATED_PATH = DATA_DIR / "curated.parquet"
DAILY_SERIES_PATH = OUTPUTS / "daily_series.parquet"
FORECAST_PATH = OUTPUTS / "forecast.parquet"
FORECAST_BACKTEST_PATH = OUTPUTS / "forecast_backtest.parquet"
FORECAST_METRICS_PATH = OUTPUTS / "forecast_metrics.json"
ALERT_SCORES_PATH = OUTPUTS / "alert_scores.parquet"
RISK_METRICS_PATH = OUTPUTS / "risk_metrics.json"
CLUSTERS_PATH = OUTPUTS / "clusters.parquet"
CLUSTER_PROFILES_PATH = OUTPUTS / "cluster_profiles.json"
CRITICAL_GROUPS_PATH = OUTPUTS / "critical_groups.parquet"
RECURRENCE_PATH = OUTPUTS / "recurrence.parquet"
ANOMALIES_PATH = OUTPUTS / "anomalies.parquet"
KPI_PROJECTION_PATH = OUTPUTS / "kpi_projection.json"
OLA_DAILY_PATH = OUTPUTS / "ola_daily.parquet"
KPI_MONTHLY_PATH = OUTPUTS / "kpi_monthly.parquet"
RECOMMENDATIONS_PATH = OUTPUTS / "recommendations.parquet"
REGISTRY_PATH = OUTPUTS / "model_registry.json"
FEATURES_PATH = OUTPUTS / "risk_features.parquet"

REF_DATE = "2025-12-31"
REGIME_BREAK = "2025-09-01"
RISK_TRAIN_END = "2025-10-31"
RISK_TEST_START = "2025-11-01"
FORECAST_BACKTEST_START = "2025-11-02"
FORECAST_BACKTEST_END = "2025-12-24"
HORIZON_DAYS = 7
COMMERCIAL_HOURS = (8, 18)

SLA_HOURS = {1: 4, 2: 4, 3: 12, 4: 24, 5: 96}

KPI_OLA_BANDS = {
    2: [  # P2 Alta
        (None, 30, 150),
        (31, 35, 125),
        (36, 39, 100),
        (40, 45, 75),
        (46, 53, 50),
        (54, None, 0),
    ],
    3: [  # P3 Média
        (None, 200, 150),
        (201, 230, 125),
        (231, 263, 100),
        (264, 290, 75),
        (291, 320, 50),
        (321, None, 0),
    ],
}

KPI_VOLUME_BANDS = {
    2: [
        (None, 4584, 150),
        (4585, 5388, 125),
        (5389, 6168, 100),
        (6169, 6252, 75),
        (6253, 6336, 50),
        (6337, None, 0),
    ],
    3: [
        (None, 19488, 150),
        (19489, 22116, 125),
        (22117, 22524, 100),
        (22525, 23892, 75),
        (23893, 24276, 50),
        (24277, None, 0),
    ],
}

LEAKAGE_COLS = frozenset(
    {
        "duracao",
        "duracao_h",
        "resolvido",
        "encerrado",
        "status",
        "codigo_fechamento",
        "solucao",
        "kpi_violado",
        "kpi_violado_flag",
        "sla_estourado",
    }
)

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
    "Código de fechamento": "codigo_fechamento",
    "Descrição resumida": "descricao",
    "Solução": "solucao",
    "Aberto por": "aberto_por",
    "Incidente Pai": "incidente_pai",
    "Status": "status",
    "Entrou para KPI?": "entrou_kpi",
    "KPI Violado?": "kpi_violado",
}

TOP_GRUPOS = 6
TOP_PRODUTOS = 5
TOP_CATEGORIAS = 5
KMEANS_K_RANGE = range(8, 21)
MONTE_CARLO_N = 10_000
RANDOM_STATE = 42
