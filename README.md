# Radar do Caos

Previsão de incidentes e priorização de alertas para operações de TI (AIOps).
Projeto acadêmico do **Enterprise Challenge FIAP × Locaweb (2026)**, desenvolvido pelo **Grupo Orango** (Tecnólogo em Ciência de Dados, FIAP).

> **Sobre os dados:** os dados de chamados vieram da Locaweb para fins educacionais e **não estão neste repositório**. O código está completo, mas o pipeline e o dashboard só rodam com a base original (ou uma base com o mesmo esquema). Modelos treinados e saídas derivadas também ficam de fora.

## O problema
Equipes de operação recebem milhares de chamados e alertas. O desafio era transformar esse histórico em inteligência preditiva: antecipar o volume de incidentes, identificar padrões e anomalias e recomendar ações para a operação.

## O que o projeto faz
- **Tratamento** (`tratamento/radar/`): pipeline modular em Python — preparo dos dados, features, previsão, risco, anomalias, padrões recorrentes, KPIs e recomendações.
- **Modelagem** (`modelagem/`): treino e comparação de modelos com holdout de 28 dias e horizonte D+1 a D+7.
  - Previsão de volume: **XGBoost** com lags, médias móveis, calendário e feriados; **Prophet** com regimes; baseline sazonal como referência.
  - Segmentação de chamados: **TF-IDF + MiniBatchKMeans**.
  - Classificador de risco de OLA: **XGBoost**.
- **Dashboard** (`dashboard/`): app **Streamlit** com páginas para antecipar incidentes, tendências, impacto em KPI e apoio à decisão.

## Resultados (holdout de 28 dias, D+1…D+7)
| Série | Modelo campeão | SMAPE |
|---|---|---|
| Volume total | XGBoost | 11,7% |
| P2 (alta) | Prophet com regimes | 45,5% |
| P3 (média) | XGBoost | 58,0% |

As séries de OLA tiveram erro percentual muito alto (SMAPE de 129% a 200%) porque têm volume baixo e muitos zeros; o erro absoluto é pequeno (RMSE 0,3 e 1,1). Detalhes e comparações em [`evidencias/`](evidencias) e [`modelagem/metrics.json`](modelagem/metrics.json).

## Estrutura
```
tratamento/   pipeline (pacote radar)
modelagem/    scripts de treino e métricas
dashboard/    app Streamlit
evidencias/   gráficos de precisão dos modelos
```

## Stack
Python, pandas, scikit-learn, XGBoost, Prophet, Streamlit, Plotly.

## Autoria
Trabalho em grupo. Integrantes: _(preencher)_.
Minha participação: _(preencher)_.
