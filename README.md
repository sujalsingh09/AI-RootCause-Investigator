# 🛒 AI Root Cause Investigator for E-commerce Operations

> An end-to-end Machine Learning and analytics system that detects unusual changes in e-commerce demand and automatically investigates the possible business factors behind those anomalies.

Built on the **Brazilian E-commerce Public Dataset by Olist**, this project simulates how an e-commerce operations team could detect an unusual demand pattern, investigate the affected business dimensions, rank possible root causes using evidence, and present the full investigation through an interactive dashboard.

Instead of just answering *"an anomaly occurred,"* the system answers:

> **"Why did this anomaly happen?"**

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Model Evaluation](#-model-evaluation)
- [Anomaly Detection](#-anomaly-detection)
- [Root-Cause Investigation](#-root-cause-investigation)
- [Evidence Scoring](#-evidence-scoring)
- [Example Investigation Walkthrough](#-example-investigation-walkthrough)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [How to Run the Project](#-how-to-run-the-project)
- [Generated Artifacts](#-generated-artifacts)
- [Model Feature Importance](#-model-feature-importance)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Production Architecture Idea](#-production-architecture-idea)
- [Why This Project?](#-why-this-project)
- [Author](#-author)
- [Disclaimer](#-disclaimer)

---

## 🎯 Problem Statement

In an e-commerce business, a sudden increase or decrease in orders can signal an important underlying event, such as:

- A product category suddenly becoming popular
- Previously inactive products selling again
- Existing products experiencing a surge in demand
- A small number of sellers driving most of the change
- Customer reviews indicating dissatisfaction
- Shifts in payment behavior
- Delivery problems affecting customer activity

The goal of this project is to **automatically detect unusual demand patterns** and **investigate the business factors** most likely associated with them.

---

## 🔍 Project Overview

The system follows an end-to-end investigation pipeline:

```
Historical E-commerce Data
          │
          ▼
   Demand Forecasting
          │
          ▼
    Anomaly Detection
          │
          ▼
 Identify Suspicious Area
          │
          ▼
Product │ Seller │ Geography
Payment │ Review │ Delivery
          │
          ▼
    Evidence Scoring
          │
          ▼
   Root-Cause Ranking
          │
          ▼
 Explainable Investigation
          │
          ▼
   Streamlit Dashboard
```

---

## ✨ Key Features

- 📈 Daily e-commerce demand forecasting (Random Forest)
- 🚨 Forecast-error based anomaly detection
- 🛡️ Data coverage issue detection (guards against false anomalies)
- 🏷️ Product category analysis
- 📦 Product behavior analysis (new / reactivated / increased)
- 🧑‍💼 Seller contribution analysis
- 🗺️ Geographic analysis
- 💳 Payment behavior analysis
- ⭐ Customer review analysis
- 📝 Review text analysis using TF-IDF
- 🚚 Delivery performance analysis
- 🧮 Evidence-based root-cause ranking
- 📊 MLflow experiment tracking
- 🖥️ Interactive Streamlit dashboard
- 📄 Explainable investigation report

---

## 📊 Dataset

This project uses the **[Brazilian E-commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)**.

The dataset contains approximately **100,000 orders** from the Brazilian e-commerce marketplace, covering orders, products, customers, sellers, payments, reviews, and delivery.

> **Note:** The dataset is historical and anonymized. This project is therefore implemented as a **prototype** of an operational root-cause investigation system rather than a live production monitoring system.

### Data Used

| Dataset | Description |
|---|---|
| `orders` | Order-level records and timestamps |
| `order_items` | Line-item level product and seller data |
| `products` | Product category and attribute data |
| `customers` | Customer and location data |
| `payments` | Payment method and value data |
| `reviews` | Review scores and text |
| `sellers` | Seller and location data |

These datasets are joined and combined during the investigation process to analyze different dimensions of e-commerce activity.

---

## 🧠 Machine Learning Pipeline

### 1. Daily Demand Creation

Order timestamps are converted into daily order counts. The system builds a continuous daily time series, filling missing calendar dates with zero orders so the forecasting model receives a consistent time-based sequence.

| Date | Orders |
|---|---|
| 2018-07-15 | 185 |
| 2018-07-16 | 192 |
| 2018-07-17 | 211 |
| 2018-07-18 | 307 |

### 2. Feature Engineering

| Feature | Description |
|---|---|
| **Lag 1** | Number of orders from the previous day |
| **Lag 7** | Number of orders from the same weekday, one week earlier |
| **Rolling 7-Day Average** | Average demand over the previous seven days (current day excluded to avoid leakage) |
| **Day of Week** | Captures weekday vs. weekend demand variation |

### 3. Demand Forecasting

A **Random Forest Regressor** is used for daily demand forecasting:

```python
RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
```

The model learns relationships between historical demand features and the number of orders expected for a given day.

### 4. Train / Test Split

A **time-based split** is used, with a forecasting cutoff of **`2018-07-01`**. Data before the cutoff is used for training; the later period serves as the holdout test set.

A time-based split is used instead of a random split because mixing historical and future observations at random would not reflect a realistic forecasting scenario.

---

## 📈 Model Evaluation

| Metric | Value | Interpretation |
|---|---|---|
| **MAE** | 24.03 | On average, predictions differ from actual demand by ~24 orders |
| **RMSE** | 52.34 | Penalizes larger prediction errors more heavily |
| **R²** | 0.817 | Share of variance explained relative to a constant baseline *(not a prediction-accuracy percentage)* |

---

## 🚨 Anomaly Detection

After generating forecasts, the system compares predicted demand against actual demand:

```
Actual Orders → Expected Orders → Forecast Error → Anomaly Score
```

```python
absolute_error = |actual - predicted|
```

Large deviations indicate that actual demand diverged meaningfully from what the model expected. A high forecast-error threshold is used to flag unusual observations.

### Data Coverage Protection

Historical datasets can have incomplete periods near their ending date. If order counts drop sharply simply because the dataset stops receiving records, this should **not** be misread as a real business collapse. The anomaly detection process therefore includes a coverage check based on recent demand patterns, preventing incomplete dataset coverage from being flagged as a genuine anomaly.

---

## 🔬 Root-Cause Investigation

Once an anomaly is detected, the system investigates it across multiple business dimensions — not just to confirm *that* demand changed, but to gather evidence for *why*:

```
Product Category → Product Behavior → Seller Contribution
→ Geography → Payment Behavior → Customer Reviews → Delivery Performance
```

### Product Category Analysis
Compares each category's share of sales on the anomaly date against a prior reference period, surfacing categories whose contribution shifted unusually (measured in percentage points).

### Product Behavior Analysis
Classifies products active around the anomaly into:

| Behavior | Definition |
|---|---|
| **New Product** | Appears in the anomaly period but wasn't seen in prior historical sales |
| **Reactivated Product** | Had historical sales, went inactive, then reappeared during the anomaly |
| **Active + Increased** | Already active, but saw a spike in activity during the anomaly |

**Example result (anomaly on 2018-07-18):**

| Behavior | Share of Items |
|---|---|
| Reactivated Product | 37.21% |
| Active + Increased | 32.56% |
| New Product | 30.23% |

This spread indicates the increase was distributed across multiple product behaviors rather than dominated by one.

### Seller Analysis
Checks whether the anomaly is concentrated among a small number of sellers.

> **Top seller contribution: 13.95%** — not high enough to be a dominant explanation, making a broad category-level demand increase more plausible than a single-seller effect.

### Geographic Analysis

| State | Share |
|---|---|
| SP | 39.74% |
| MG | 13.36% |
| RJ | 12.05% |
| SC | 4.23% |
| BA | 4.23% |

São Paulo held the largest share of anomaly-day orders. Geographic concentration alone doesn't prove causation, so it's treated as a **supporting signal**.

### Payment Analysis

| Method | Share |
|---|---|
| Credit Card | 71.83% |
| Boleto | 17.03% |
| Voucher | 6.50% |
| Debit Card | 4.64% |

No unusual shift in payment behavior was detected — ranked as a **weak signal**.

### Customer Review Analysis

- Average review score: **4.49 / 5**
- Low score rate: **5.75%**

Not unusually elevated compared with the reference period — dissatisfaction was **not** a primary driver.

### Review Text Analysis
Review comments are analyzed with **TF-IDF** to surface frequently occurring terms and themes (e.g., *produto, entrega, recomendo, prazo, chegou*). Review text in the dataset is primarily **Portuguese**. These terms are treated as textual signals, not proof of causation.

### Delivery Analysis

- Average delivery time: **8.76 days**
- Late delivery rate: **2.93%**

Delivery performance did not provide strong evidence of an operational problem behind the anomaly.

---

## 🧮 Evidence Scoring

A heuristic scoring system combines all investigation signals into a ranked, interpretable explanation:

| Signal | Evidence Strength |
|---|---|
| Category demand surge | 🟢 Primary |
| Reactivated products | 🟡 Supporting |
| Active + increased products | 🟡 Supporting |
| New products | 🟡 Supporting |
| Seller concentration | ⚪ Weak |
| Payment behavior | ⚪ Weak |
| Customer dissatisfaction | ⚫ Context |
| Delivery problems | ⚫ Context |

> ⚠️ **Important:** These scores are heuristic signals. They do **not** represent statistical probabilities and do **not** prove causation.

---

## 🕵️ Example Investigation Walkthrough

**Investigated date:** `2018-07-18`

| Metric | Value |
|---|---|
| Actual orders | 307 |
| Expected orders | 211.91 |
| Demand lift | **+44.87%** |
| Anomaly detected | ✅ True |

**Strongest category signal:** `cama_mesa_banho`, with share increasing by **+4.04 percentage points** vs. the reference period.

**Product-level evidence:**
- Reactivated products: 37.21%
- Active + increased: 32.56%
- New products: 30.23%

**Seller evidence:** Largest seller contributed 13.95% — no dominant single-seller effect.

**Customer evidence:** 4.49 / 5 average score, 5.75% low-score rate — no strong dissatisfaction signal.

**Delivery evidence:** 8.76-day average delivery, 2.93% late-delivery rate — not a likely driver.

### Conclusion

The strongest evidence points to a **broad demand surge in the `cama_mesa_banho` category**, distributed across reactivated products, existing products with increased activity, and new products. Seller concentration, payment behavior, customer dissatisfaction, and delivery performance all provided weaker supporting evidence.

**Final ranking:** *Broad category demand surge* is the most likely explanation — presented as an **evidence-based hypothesis**, not proof of causation.

---

## 🖥️ Streamlit Dashboard

An interactive dashboard lets users:

- Select an investigation date
- View actual vs. expected demand
- Inspect anomaly scores and top anomalies
- Analyze category contribution
- Inspect product behavior
- Review seller contribution
- Explore geographic distribution
- Analyze payment methods
- Inspect review metrics and text signals
- Review delivery performance
- Read the final root-cause conclusion

Designed to be understandable by both technical and business users.

---

## 🧪 MLflow Experiment Tracking

**MLflow** tracks the forecasting experiment, recording run parameters, metrics, and artifacts locally — making the model development process reproducible and easy to inspect.

---

## 📁 Project Structure

```
AI-RootCause-Investigator/
│
├── app/
│   └── dashboard.py
│
├── artifacts/
│   ├── models/
│   └── feature_importance.csv
│
├── data/
│   └── raw/
│
├── notebooks/
│
├── src/
│   ├── data_loader.py
│   ├── forecasting.py
│   ├── anomaly_detection.py
│   ├── product_analysis.py
│   └── investigation.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Technology Stack

| Category | Tools |
|---|---|
| **Programming** | Python, Pandas, NumPy |
| **Machine Learning** | Scikit-learn, Random Forest, TF-IDF |
| **Model & Experiment Management** | MLflow, Joblib |
| **Visualization** | Matplotlib, Seaborn |
| **Dashboard** | Streamlit |

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI-RootCause-Investigator
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Add the Dataset

Download the Olist Brazilian E-commerce dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place the CSV files inside:

```
data/raw/
```

> Raw CSV files are intentionally excluded from Git via `.gitignore`.

### 6. Run the ML Pipeline

```bash
python main.py
```

The pipeline will:

```
Load Data → Create Daily Demand → Train Forecasting Model → Evaluate Model
→ Detect Anomalies → Investigate Root Cause → Track Experiment → Save Model Artifacts
```

### 7. Start the Dashboard

```bash
streamlit run app/dashboard.py
```

The dashboard will open automatically in your browser.

---

## 📦 Generated Artifacts

```
artifacts/
│
├── models/
│   └── demand_forecasting_model.pkl
│
└── feature_importance.csv
```

- The trained forecasting model is saved using **Joblib**.
- Feature importance is saved separately so the model's most influential features can be inspected.

---

## 🔑 Model Feature Importance

The forecasting model currently uses:

- `lag_1`
- `lag_7`
- `rolling_7`
- `day_of_week`

The strongest predictor is **`lag_1`** (previous day's demand), followed by the 7-day rolling average and 7-day lag — indicating recent demand history is highly informative for short-term order volume forecasting.

---

## ⚠️ Limitations

- **Historical data** — the system doesn't monitor a live e-commerce platform.
- **Dataset coverage** — incomplete data near the dataset's ending period is mitigated with a dedicated coverage-protection check.
- **Heuristic evidence scores** — root-cause scores are heuristic, not statistical probabilities.
- **Causality** — the system surfaces evidence *consistent with* possible explanations; it does not prove causation.
- **Forecasting quality** — depends on the historical patterns available in the dataset.
- **Review language** — review text is primarily Portuguese; text analysis is performed in the original language.

---

## 🔮 Future Improvements

- Real-time order data integration
- Automated anomaly alerts
- More advanced forecasting models (XGBoost, gradient boosting)
- Time-series-specific forecasting models
- SHAP-based model explanations
- Statistical significance testing
- Causal inference
- LLM-generated investigation summaries
- Real-time streaming pipelines
- Automated model retraining
- Cloud deployment
- API-based architecture
- Role-based dashboard access
- Automated email / Slack alerts

---

## 🏗️ Production Architecture Idea

```
Live E-commerce Data
        │
        ▼
 Data Ingestion Layer
        │
        ▼
 Feature Engineering
        │
        ▼
 Forecasting Service
        │
        ▼
 Anomaly Detection
        │
        ▼
 Root Cause Investigation
        │
        ▼
 Evidence Ranking
        │
        ▼
 Alert / API / Dashboard
```

The current project focuses on building and validating the core investigation workflow using historical data as a foundation for this architecture.

---

## 💡 Why This Project?

Traditional anomaly detection systems typically stop at:

> *"Something unusual happened."*

This project goes one step further:

> *"Something unusual happened — and here are the business signals that may explain it."*

Combining forecasting, anomaly detection, multi-dimensional analysis, evidence scoring, and explainable reporting makes this a useful prototype for operational analytics and AI-assisted business investigation.

---

## 👤 Author

**Sujal Singh Jhala**
B.Tech Electronics Engineering, Madhav Institute of Technology & Science, Gwalior

Interested in: Machine Learning · Artificial Intelligence · NLP · Data Science · MLOps · Software Engineering

---

## 📄 Disclaimer

This project is an analytical prototype built using a public historical dataset. Root-cause findings should be interpreted as **evidence-based hypotheses**, not guaranteed causal explanations. The architecture is designed so the historical dataset can eventually be replaced with live business data in a production environment.