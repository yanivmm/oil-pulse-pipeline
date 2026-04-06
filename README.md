# 🛢️ Oil Pulse Pipeline

**A production-style data pipeline that tracks crude oil prices, Reddit sentiment, and news signals to predict next-day oil price direction (Up / Down).** Built with Apache Airflow for orchestration, PySpark for transformations, DuckDB for storage, and Streamlit for visualization. Designed as a portfolio project demonstrating real-world data engineering patterns.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AIRFLOW  ORCHESTRATION                          │
│                                                                        │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────┐   │
│  │ DAG 1        │    │ DAG 2            │    │ DAG 3              │   │
│  │ INGEST       │───▶│ TRANSFORM        │───▶│ PREDICT & PUBLISH  │   │
│  │ (hourly)     │    │ (daily)          │    │ (daily)            │   │
│  └──────┬───────┘    └──────┬───────────┘    └──────┬─────────────┘   │
│         │                   │                       │                  │
│         ▼                   ▼                       ▼                  │
│  ┌─────────────┐    ┌─────────────────┐    ┌────────────────────┐     │
│  │ FileSensor   │    │ ExternalTask    │    │ ExternalTask       │     │
│  │ PythonOp x3 │    │ BranchPython    │    │ ShortCircuitOp     │     │
│  │ XCom        │    │ TaskGroup       │    │ SLA + Callbacks    │     │
│  │ TriggerRule │    │ SparkSubmit     │    │ EmailOp (mock)     │     │
│  └─────────────┘    └─────────────────┘    └────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘

DATA FLOW:

  Yahoo Finance ─┐                        ┌─► /data/processed/clean/
  Reddit API ────┼─► /data/raw/ ──SPARK──▶├─► /data/processed/features/
  RSS Feeds ─────┘                        ├─► /data/processed/aggregated/
                                          └─► /data/processed/predictions/
                                                       │
                                                       ▼
                                               ┌──────────────┐
                                               │   DuckDB      │
                                               │ oil_pulse.db  │
                                               └──────┬───────┘
                                                      │
                                                      ▼
                                               ┌──────────────┐
                                               │  Streamlit    │
                                               │  Dashboard    │
                                               └──────────────┘
```

---

## Tech Stack

| Layer              | Technology                        | Purpose                              |
|--------------------|-----------------------------------|--------------------------------------|
| Orchestration      | Apache Airflow 2.x (Astronomer)   | DAG scheduling, sensors, branching   |
| Transformation     | PySpark (local mode)              | Cleaning, joins, window functions    |
| Ingestion          | yfinance, praw, feedparser        | Oil prices, Reddit, RSS news         |
| NLP                | VADER Sentiment                   | Sentiment scoring                    |
| ML                 | scikit-learn (RandomForest)       | Price direction classification       |
| Storage            | DuckDB                            | Lightweight analytical database      |
| Visualization      | Streamlit + Plotly                 | Interactive dashboard                |
| Testing            | pytest                            | Spark transforms + DAG validation    |
| Containerization   | Docker + docker-compose           | Reproducible local environment       |

---

## Folder Structure

```
oil-pulse-pipeline/
├── dags/
│   ├── dag_ingest.py              # DAG 1: hourly ingestion from 3 sources
│   ├── dag_transform.py           # DAG 2: Spark clean → features → aggregate
│   └── dag_predict_publish.py     # DAG 3: ML predict → DuckDB → notify
├── scripts/
│   ├── ingest/
│   │   ├── fetch_oil_prices.py    # Yahoo Finance CL=F crude oil futures
│   │   ├── fetch_reddit_sentiment.py  # Reddit + VADER sentiment
│   │   └── fetch_rss_news.py      # RSS feeds + VADER sentiment
│   ├── transform/
│   │   ├── spark_clean.py         # Schema enforcement, nulls, dedup
│   │   ├── spark_features.py      # Window functions, joins, labels
│   │   └── spark_aggregate.py     # Daily rollups for dashboard
│   └── ml/
│       ├── train_model.py         # RandomForest classifier
│       ├── predict.py             # Daily prediction
│       └── load_duckdb.py         # Upsert into DuckDB
├── dashboard/
│   └── app.py                     # Streamlit dashboard
├── tests/
│   ├── test_dag_structure.py      # DAG import + dependency checks
│   ├── test_spark_clean.py        # Spark cleaning logic
│   └── test_spark_features.py     # Window functions + joins
├── data/
│   ├── raw/                       # Landing zone (CSV per source per day)
│   └── processed/                 # Spark output (Parquet + CSV)
├── models/                        # Serialized ML models
├── docker-compose.override.yml    # Volume mounts + resource limits
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
└── README.md                      # You are here
```

---

## Setup Instructions

> **Prerequisites:** Windows 10/11, Docker Desktop, Python 3.10+

### Step 1 — Clone the repo
```bash
git clone https://github.com/<your-username>/oil-pulse-pipeline.git
cd oil-pulse-pipeline
```

### Step 2 — Install Docker Desktop
Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) and enable **WSL 2 backend** during setup. Restart your machine after install.

### Step 3 — Install Astronomer Astro CLI
```bash
# In WSL2 or PowerShell
curl -sSL install.astronomer.io | sudo bash -s
```
Or on Windows with winget:
```powershell
winget install -e --id Astronomer.Astro
```

### Step 4 — Create environment file
```bash
cp .env.example .env
# Edit .env and fill in your Reddit API credentials (see .env.example)
```

### Step 5 — Install Python dependencies (for local development)
```bash
pip install -r requirements.txt
```

### Step 6 — Start Airflow locally
```bash
astro dev start
```
Airflow UI will be available at **http://localhost:8080** (user: `admin`, password: `admin`).

### Step 7 — Run the pipeline
Enable all three DAGs in the Airflow UI. `dag_ingest` runs hourly; the other two are triggered daily. You can also trigger manually from the UI.

### Step 8 — Run the dashboard
```bash
streamlit run dashboard/app.py
```
Dashboard will be available at **http://localhost:8501**.

### Step 9 — Run tests
```bash
pytest tests/ -v
```

### Step 10 — Run scripts standalone (optional)
Every script can run independently for debugging:
```bash
python scripts/ingest/fetch_oil_prices.py
python scripts/transform/spark_clean.py
python scripts/ml/train_model.py
```

---

## How to Run the Dashboard

```bash
# Make sure the pipeline has run at least once to populate DuckDB
streamlit run dashboard/app.py
```

The dashboard reads directly from `data/oil_pulse.duckdb`. If the database doesn't exist yet, it shows a friendly placeholder message prompting you to run the pipeline first.

---

## Screenshots

> 📸 Screenshots coming soon — will include Airflow DAG graph views and Streamlit dashboard.

---

## Key Concepts Demonstrated

- **Airflow:** FileSensor, ExternalTaskSensor, BranchPythonOperator, ShortCircuitOperator, TaskGroup, XCom, TriggerRules, SLA callbacks, on_failure_callback, EmailOperator (mocked)
- **Spark:** Explicit schema definitions (StructType), window functions (rolling avg, lag), joins, null handling, deduplication, Parquet I/O
- **Data Engineering:** Multi-source ingestion, batch processing, feature engineering, idempotent loads (upsert), local-first architecture
- **ML:** Binary classification, train/predict split, model serialization, prediction logging
- **Testing:** Spark unit tests with local SparkSession, DAG structure validation with DagBag

---

## License

MIT
