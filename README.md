# 🌍 HumaCrisis Forecasting System

> **Multi-dimensional humanitarian crisis forecasting for Sub-Saharan Africa**  
> Combining food insecurity, conflict, and displacement data to predict crisis severity 3 months ahead.

[![CI/CD](https://github.com/TON_USERNAME/humacrisis-forecasting-system/actions/workflows/ci.yml/badge.svg)](https://github.com/TON_USERNAME/humacrisis-forecasting-system/actions)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.9-orange)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 Overview

HumaCrisis is an end-to-end machine learning system that forecasts humanitarian crisis severity across 10 Sub-Saharan African countries. It integrates three critical data dimensions:

| Dimension | Source | Indicators |
|---|---|---|
| 🌾 Food Insecurity | FEWS NET | IPC Phase, Population Affected |
| ⚔️ Conflict | ACLED | Events, Fatalities, Battle Types |
| 🚶 Displacement | UNHCR | IDPs, Refugees, Returnees |

**Target countries:** Somalia, South Sudan, Ethiopia, CAR, DR Congo, Mali, Niger, Chad, Mauritania, Senegal

---

## 🏗️ Architecture
```
humacrisis-forecasting-system/
├── src/
│   ├── ingestion/          # Data collection pipelines
│   │   ├── fewsnet_collector.py
│   │   ├── acled_collector.py
│   │   ├── unhcr_collector.py
│   │   └── pipeline.py
│   ├── features/           # Feature engineering
│   │   └── feature_engineering.py
│   ├── models/             # ML models
│   │   ├── xgboost_model.py
│   │   ├── lstm_model.py
│   │   └── ensemble_model.py
│   ├── api/                # FastAPI REST API
│   │   └── main.py
│   ├── dashboard/          # Streamlit dashboard
│   │   └── app.py
│   └── utils/
│       └── logger.py
├── data/
│   ├── raw/                # Raw data per source
│   └── processed/          # Processed datasets
├── models/trained/         # Saved ML models
├── reports/figures/        # EDA visualizations
├── tests/                  # Unit tests
├── .github/workflows/      # CI/CD pipeline
├── Dockerfile
└── docker-compose.yml
```

---

## 🚀 Quick Start

### 1 — Clone & Setup
```bash
git clone https://github.com/TON_USERNAME/humacrisis-forecasting-system.git
cd humacrisis-forecasting-system
conda create -n humacrisis python=3.10 -y
conda activate humacrisis
pip install -r requirements.txt
```

### 2 — Run Data Pipeline
```bash
export PYTHONPATH=$PWD
python -m src.ingestion.pipeline
```

### 3 — Feature Engineering
```bash
python -m src.features.feature_engineering
```

### 4 — Train Models
```bash
python -m src.models.ensemble_model
```

### 5 — Launch API
```bash
uvicorn src.api.main:app --reload --port 8000
```

### 6 — Launch Dashboard
```bash
streamlit run src/dashboard/app.py
```

### 7 — Docker (all-in-one)
```bash
docker-compose up --build
```

---

## 📊 Model Performance

| Model | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost | ~0.045 | ~0.062 | ~0.87 |
| LSTM + Attention | ~0.052 | ~0.071 | ~0.83 |
| **Ensemble** | **~0.041** | **~0.058** | **~0.89** |

*Evaluated with 5-fold time-series cross-validation*

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/countries` | List all countries |
| GET | `/status/{country}` | Current country status |
| POST | `/predict` | Generate crisis forecast |

### Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "country": "SOM",
    "date": "2024-01-01",
    "ipc_phase": 3.5,
    "total_events": 80,
    "total_fatalities": 400,
    "total_displaced": 2900000,
    "forecast_horizon": 3
  }'
```

### Example Response
```json
{
  "country": "SOM",
  "forecast_date": "2024-04-01",
  "crisis_severity_index": 0.7234,
  "risk_level": "HIGH",
  "confidence": 0.87,
  "recommendations": [
    "Scale up humanitarian operations in SOM",
    "Pre-position emergency supplies",
    "Strengthen early warning monitoring"
  ]
}
```

---

## 📈 Key Features

- **71 engineered features** including lag variables, rolling statistics, and interaction terms
- **Seasonal features** capturing lean season and harvest cycles in Sub-Saharan Africa
- **SHAP explainability** for model interpretability
- **MLflow tracking** for experiment management
- **Time-series cross-validation** to prevent data leakage
- **CI/CD pipeline** with GitHub Actions
- **Docker containerization** for reproducibility

---

## 🌍 Dashboard Screenshots

| Overview | Country Analysis |
|---|---|
| KPI cards + Crisis timeline | Multi-indicator country drill-down |

| Forecasting | Model Insights |
|---|---|
| Interactive gauge + Recommendations | Feature importance + Training curves |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Data** | Pandas, NumPy, GeoPandas |
| **ML** | XGBoost, PyTorch (LSTM), Scikit-learn, SHAP |
| **Tracking** | MLflow |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Dashboard** | Streamlit, Plotly |
| **DevOps** | Docker, GitHub Actions |
| **Quality** | Pytest, Black, Flake8 |

---

## 👤 Author

**Serge Nyamsin** — Data Scientist | Humanitarian & Development Specialist  
12+ years field experience with French Red Cross, Action Against Hunger, Handicap International  
MSc Data Science & AI — DSTI

[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-black)](https://github.com/sanyamsin)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/www.linkedin.com/in/serge-alain-nyamsin)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

