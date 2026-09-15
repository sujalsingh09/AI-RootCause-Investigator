# 🛒 AI Root Cause Investigator for E-commerce Operations

> An end-to-end Machine Learning and analytics system that forecasts e-commerce demand, detects unusual demand patterns, and investigates the business factors that may be associated with those changes.

Built using the **Brazilian E-commerce Public Dataset by Olist**, this project demonstrates how an e-commerce operations team could combine demand forecasting, anomaly detection, and multi-dimensional business analysis to understand unusual changes in order volume.

Instead of stopping at:

> **"Something unusual happened."**

the system aims to answer:

> **"What business signals may explain this unusual change?"**

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Feature Engineering](#-feature-engineering)
- [Demand Forecasting](#-demand-forecasting)
- [Time-Based Train/Test Split](#-time-based-traintest-split)
- [Model Evaluation](#-model-evaluation)
- [Anomaly Detection](#-anomaly-detection)
- [Historical vs Future Analysis](#-historical-vs-future-analysis)
- [Root-Cause Investigation](#-root-cause-investigation)
- [Evidence-Based Investigation](#-evidence-based-investigation)
- [Example Investigation](#-example-investigation)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [How to Run the Project](#-how-to-run-the-project)
- [Model Feature Importance](#-model-feature-importance)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Production Architecture Idea](#-production-architecture-idea)
- [Why This Project](#-why-this-project)
- [Author](#-author)
- [Disclaimer](#-disclaimer)

---

## 🎯 Problem Statement

In an e-commerce business, changes in daily order volume can contain important operational information.

A sudden increase or decrease in demand may be associated with:

- Changes in product-category demand
- Changes in product activity
- Seller-level concentration
- Geographic demand patterns
- Payment behavior
- Customer review signals
- Delivery performance
- Other changes in the marketplace

A forecasting model can estimate how many orders would normally be expected. The next step is to compare the expected demand with the actual demand and investigate the business dimensions associated with the difference.

The goal of this project is therefore to build a system that can:

1. Forecast daily order demand
2. Compare expected demand with actual demand
3. Detect unusually large deviations
4. Investigate possible contributing business factors
5. Provide an explainable result through an interactive dashboard

---

## 🔍 Project Overview

The current system follows this workflow:

```text
                 User Selects Date
                        │
                        ▼
                Demand Forecasting
                        │
                        ▼
                 Expected Demand
                        │
                        ▼
              Is Actual Data Reliable?
                   /            \
                 NO              YES
                 │                │
                 ▼                ▼
          Forecast Only     Compare Actual
                            vs Forecast
                                │
                                ▼
                         Anomaly Detection
                           /          \
                         NO            YES
                         │              │
                         ▼              ▼
                       Normal       Investigation
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
                 Product             Seller             Geography
                    │                   │                   │
                    ├──────────────┬────┴──────────────┐    │
                    ▼              ▼                   ▼    ▼
                Payment         Reviews            Delivery
                    │              │                   │
                    └──────────────┴───────────────────┘
                                   │
                                   ▼
                         Likely Contributing Factors
                                   │
                                   ▼
                           Streamlit Dashboard
```

---

## ✨ Key Features

### Forecasting
- 📈 Daily e-commerce demand forecasting
- 🌲 Random Forest regression model
- 🕐 Time-based train/test split
- 🔢 Lag-based demand features
- 📊 Rolling demand features
- 📅 Day-of-week feature
- 🔮 Recursive forecasting for future dates

### Anomaly Detection
- 🚨 Forecast-error based anomaly detection
- 📊 Actual vs predicted demand comparison
- 📈 Absolute forecast-error analysis
- 🛡️ Protection against incomplete dataset coverage near the end of the historical data

### Business Investigation
- 🏷️ Product category analysis
- 📦 Product behavior analysis
- 🧑‍💼 Seller contribution analysis
- 🗺️ Geographic analysis
- 💳 Payment analysis
- ⭐ Customer review analysis
- 📝 Review text analysis
- 🚚 Delivery performance analysis

### Dashboard
- 🖥️ Interactive Streamlit dashboard
- 📅 Date-based investigation
- 📈 Forecast visualization
- 🚨 Anomaly identification
- 🔍 Multi-dimensional investigation
- 📊 Explainable investigation results

---

## 📊 Dataset

This project uses the **[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)**.

The dataset contains approximately **100,000 anonymized orders** from the Brazilian e-commerce marketplace, including information related to:

- Orders
- Customers
- Sellers
- Products
- Payments
- Reviews
- Order items
- Product categories
- Geographic information

> The dataset is historical and anonymized. Therefore, this project is implemented as a **prototype** of an e-commerce operational analytics and investigation system rather than a live production monitoring platform.

### 📂 Data Used

| Dataset | Purpose |
|---|---|
| `olist_orders_dataset.csv` | Order timestamps and order status |
| `olist_order_items_dataset.csv` | Product and seller activity |
| `olist_products_dataset.csv` | Product and category information |
| `olist_customers_dataset.csv` | Customer and location information |
| `olist_sellers_dataset.csv` | Seller information |
| `olist_order_payments_dataset.csv` | Payment methods and payment values |
| `olist_order_reviews_dataset.csv` | Customer review scores and text |
| `olist_geolocation_dataset.csv` | Geographic information |
| `product_category_name_translation.csv` | Product category translation |

> Raw dataset files are kept locally and are **not** intended to be committed to GitHub.

---

## 🧠 Machine Learning Pipeline

The forecasting pipeline converts order-level data into a daily demand time series:

```text
Raw Orders
    │
    ▼
Convert Purchase Timestamp
    │
    ▼
Daily Order Count
    │
    ▼
Continuous Daily Time Series
    │
    ▼
Feature Engineering
    │
    ├── Lag 1
    ├── Lag 7
    ├── Rolling 7
    └── Day of Week
    │
    ▼
Time-Based Train/Test Split
    │
    ▼
Random Forest Regressor
    │
    ▼
Demand Forecast
    │
    ▼
Forecast Evaluation
    │
    ▼
Anomaly Detection
```

### ⚙️ Feature Engineering

The forecasting model currently uses four main features:

| Feature | Description |
|---|---|
| `lag_1` | Number of orders on the previous day |
| `lag_7` | Number of orders seven days earlier |
| `rolling_7` | Average order volume over the previous seven days |
| `day_of_week` | Day of the week represented as an integer from 0–6 |

**Example** — for a particular date:

```text
Previous Day Orders      → lag_1
Orders 7 Days Earlier    → lag_7
Previous 7-Day Average   → rolling_7
Day of Week              → day_of_week
```

> The rolling feature uses previous observations and excludes the current day's order count to avoid data leakage.

### 📈 Demand Forecasting

A **Random Forest Regressor** is used to forecast daily order demand.

Current configuration:

```python
RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
```

The model learns the relationship between historical demand patterns and the number of orders expected for a given day.

### 🕐 Time-Based Train/Test Split

The project uses a **chronological split** rather than a random split.

Current forecasting cutoff: **`2018-07-01`**

- Data before the cutoff → used for training
- Data from the cutoff onward → used as the holdout test period

```text
Historical Data
      │
      ├─────────────── Training ───────────────┐
      │                                        │
      │                                  2018-07-01
      │                                        │
      └──────────────── Test ──────────────────┘
```

A time-based split is important for forecasting because future information should not be randomly mixed into the training data.

---

## 📊 Model Evaluation

The current Random Forest forecasting model achieved the following results on the holdout test period:

| Metric | Value |
|---|---|
| MAE | 24.03 |
| RMSE | 52.34 |
| R² | 0.818 |

**Metric meaning:**

- **MAE (Mean Absolute Error)** — the model's predictions differ from actual demand by approximately 24 orders on average.
- **RMSE (Root Mean Squared Error)** — gives more weight to larger prediction errors.
- **R² (R-squared)** — approximately 0.818, indicating the model explains a substantial portion of the variation in the holdout data relative to a constant baseline. This should **not** be interpreted as "81.8% prediction accuracy."

---

## 🚨 Anomaly Detection

The system compares actual order demand with the forecasted demand:

```text
Actual Demand
      │
      ▼
Expected Demand
      │
      ▼
Actual - Expected
      │
      ▼
Forecast Error
      │
      ▼
Absolute Error
      │
      ▼
Anomaly Threshold
```

The anomaly score is based on the magnitude of the forecast error:

```python
absolute_error = abs(actual - predicted)
```

Large deviations between actual and expected demand are flagged for investigation.

### 🛡️ Dataset Coverage Protection

The Olist dataset contains a sparse/incomplete tail near its final date. The dataset extends into October 2018, but the later portion does not provide a reliable representation of complete daily business activity.

Therefore, the system uses a reliable actual-data cutoff: **`2018-09-01`**

Dates after this point are **not** automatically treated as real business anomalies — this prevents incomplete historical coverage from being interpreted as an actual collapse in e-commerce demand.

---

## 🔮 Historical vs Future Analysis

The system handles historical and future dates differently.

### Historical Date

If the selected date is within the reliable historical period:

```text
Selected Date
      │
      ▼
Actual Available
      │
      ▼
Forecast
      │
      ▼
Actual vs Forecast
      │
      ▼
Anomaly?
```

- If the difference is normal → **Normal Demand**
- If the difference is unusually large:

```text
Anomaly Detected
        │
        ▼
Root-Cause Investigation
```

### Future / Unreliable Date

For dates after the reliable actual-data cutoff:

```text
Selected Future Date
        │
        ▼
Recursive Forecast
        │
        ▼
Expected Demand
        │
        ▼
Forecast Only
```

Since reliable actual demand is unavailable, the system does not attempt to classify the future date as an anomaly or perform root-cause analysis.

> This distinction matters: **a forecast is not the same thing as an actual business observation.**

---

## 🔬 Root-Cause Investigation

When an unusual historical demand pattern is detected, the system investigates multiple business dimensions. The investigation is designed to identify **likely contributing factors**, rather than claiming statistical causation.

```text
Anomaly
   │
   ├── Product Categories
   │
   ├── Product Behavior
   │
   ├── Sellers
   │
   ├── Geography
   │
   ├── Payments
   │
   ├── Customer Reviews
   │
   └── Delivery Performance
```

### 🏷️ Product Category Analysis

The system compares category activity on the investigated date against a reference period, helping identify categories whose share of orders changed significantly.

```text
Category
    │
    ├── Historical Share
    │
    └── Investigation-Day Share
             │
             ▼
       Share Difference
```

Categories with larger changes become stronger investigation signals.

### 📦 Product Behavior Analysis

Products are analyzed based on their recent and historical activity:

| Behavior | Meaning |
|---|---|
| **New Product** | Activity appears in the investigation period but was not observed in the earlier reference period |
| **Reactivated Product** | Had historical activity, became inactive, and appeared again |
| **Active + Increased** | Already active, but experienced increased activity |

These classifications help determine whether a demand change was broad across existing products or concentrated in particular product behaviors.

### 🧑‍💼 Seller Analysis

Seller activity is analyzed to determine whether unusual demand is concentrated among a small number of sellers. The investigation can surface:

- Top sellers by order contribution
- Seller-level changes compared with a reference period
- Sellers contributing to unusual demand changes

A high concentration may indicate that a small number of sellers are associated with the observed demand change. However, **seller concentration is treated as evidence associated with the anomaly, not proof that the seller caused it.**

### 🗺️ Geographic Analysis

Customer and seller location information can be used to understand geographic distribution. The analysis can identify:

- States with high order contribution
- Geographic concentration
- Changes in geographic demand patterns

Geographic concentration is treated as a **supporting signal** rather than direct causal evidence.

### 💳 Payment Analysis

Payment data is analyzed to understand whether the anomaly is associated with changes in payment behavior. The investigation considers:

- Payment method
- Number of transactions
- Payment value
- Distribution of payment methods

Example payment methods include Credit Card, Boleto, Debit Card, and Voucher.

If payment behavior does not change significantly, it becomes a weaker explanation for the observed anomaly.

### ⭐ Customer Review Analysis

Customer review information can provide additional business context. The system can examine:

- Review scores
- Average review score
- Distribution of review ratings
- Low-score reviews

Review data is treated as **supporting evidence**. An important limitation is that reviews may be submitted after the original purchase, so same-day review activity should not automatically be interpreted as the cause of same-day demand.

### 📝 Review Text Analysis

Review comments can also be analyzed to identify recurring terms and themes. The dataset contains primarily **Portuguese** review text.

Text analysis can help surface recurring terms related to topics such as:

```text
produto
entrega
recomendo
prazo
chegou
```

These terms are useful as contextual signals but should **not** be interpreted as proof of causation.

### 🚚 Delivery Performance Analysis

Delivery information can be analyzed to understand operational performance. The investigation considers:

- Average delivery time
- Median delivery time
- Delivery completion
- Late delivery behavior

Delivery metrics can provide additional evidence about whether operational problems may be associated with unusual customer/order behavior.

---

## 🧮 Evidence-Based Investigation

The investigation combines multiple signals to form an interpretable explanation:

```text
Forecast Anomaly
      │
      ├── Category Signal
      ├── Product Signal
      ├── Seller Signal
      ├── Geographic Signal
      ├── Payment Signal
      ├── Review Signal
      └── Delivery Signal
              │
              ▼
      Evidence Comparison
              │
              ▼
    Likely Contributing Factors
```

The system should be interpreted as an **evidence-ranking framework**. It does not claim:

> "This factor definitely caused the anomaly."

Instead, it answers:

> "This factor shows stronger evidence associated with the observed anomaly than the other investigated factors."

---

## 🕵️ Example Investigation

One investigated historical date is **`2018-07-19`**.

The forecasting model produced a large deviation between expected and actual demand:

| Metric | Value |
|---|---|
| Actual Orders | 253 |
| Predicted Orders | 673.71 |
| Absolute Error | 420.71 |
| Anomaly | Yes |

The model significantly **overestimated** demand on this date.

This example is particularly useful because it demonstrates an important distinction: **a large forecasting error does not automatically mean the business experienced a true demand anomaly.**

The date can be flagged for investigation based on forecast deviation, but the investigation should consider whether the difference represents genuine business behavior or simply model error.

For example, the model relied heavily on recent demand features such as:

- Previous-day demand
- Seven-day rolling demand
- Seven-day lag

The unusually high demand on the previous day caused the forecasting model to expect a much larger number of orders on the following day.

Therefore, the observed deviation is evidence of **forecast error**, not automatically proof of an underlying business event.

---

## 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard that allows users to:

- Select an investigation date
- View expected demand
- View actual demand when reliable actual data is available
- Compare actual vs predicted demand
- Identify anomalies
- Investigate product categories
- Analyze product behavior
- Analyze seller contribution
- Explore geographic patterns
- Inspect payment behavior
- Review customer review information
- Examine delivery performance
- View investigation results

For dates after the reliable historical-data cutoff, the dashboard switches to **Forecast Only** mode and does not perform historical anomaly/root-cause analysis.

---

## 📁 Project Structure

```text
AI-RootCause-Investigator/
│
├── app/
│   └── dashboard.py
│
├── artifacts/
│
├── data/
│   └── raw/
│
├── notebooks/
│   └── 01_data_understanding.ipynb
│
├── screenshots/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── forecasting.py
│   ├── anomaly_detection.py
│   ├── investigation.py
│   └── product_analysis.py
│
├── main.py
├── mlflow
├── README.md
├── requirements.txt
└── .gitignore
```

> `data/raw/` contains the locally downloaded Olist dataset and should **not** be committed to the repository.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Forecasting Model | Random Forest Regressor |
| Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Development | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

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

### 5. Download the Dataset

Download the **[Olist Brazilian E-commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)** and place the CSV files inside `data/raw/`.

The expected directory structure:

```text
data/
└── raw/
    ├── olist_orders_dataset.csv
    ├── olist_order_items_dataset.csv
    ├── olist_products_dataset.csv
    ├── olist_customers_dataset.csv
    ├── olist_sellers_dataset.csv
    ├── olist_order_payments_dataset.csv
    ├── olist_order_reviews_dataset.csv
    ├── olist_geolocation_dataset.csv
    └── product_category_name_translation.csv
```

### 6. Run the Main Pipeline

```bash
python main.py
```

### 7. Start the Streamlit Dashboard

```bash
streamlit run app/dashboard.py
```

The dashboard should then open in your browser.

---

## 🔑 Model Feature Importance

The forecasting model currently uses:

- `lag_1`
- `lag_7`
- `rolling_7`
- `day_of_week`

The model's feature importance provides insight into which historical demand signals are most useful for the Random Forest model. In the current experiment, the strongest features (in order) were:

1. `lag_1`
2. `rolling_7`
3. `lag_7`
4. `day_of_week`

This indicates that recent demand history is particularly informative for short-term order forecasting.

---

## ⚠️ Limitations

**1. Historical Dataset**
The project uses historical Olist data rather than a live e-commerce data stream.

**2. Dataset Coverage**
The later portion of the dataset contains incomplete/sparse coverage. A reliable actual-data cutoff is therefore used to prevent false anomaly detection.

**3. Forecasting Model**
The current forecasting model is a Random Forest regression model using a small set of time-based features. More advanced time-series methods could potentially improve forecasting performance.

**4. Forecast Error vs Business Anomaly**
A large difference between actual and predicted demand can be caused by:

- Genuine business behavior
- Unexpected demand
- Model limitations
- Insufficient features
- Sudden events not represented in historical data

Therefore, **forecast error should not automatically be interpreted as a business anomaly.**

**5. Root-Cause Analysis**
The investigation identifies likely contributing factors and supporting evidence. It does **not** establish statistical or causal relationships.

**6. Review Timing**
Customer reviews may occur after the purchase date, so review information should be interpreted carefully.

**7. Review Language**
The review text is primarily Portuguese, which can limit the effectiveness of simple English-oriented text-analysis approaches.

---

## 🔮 Future Improvements

**Forecasting**
- XGBoost forecasting
- Gradient Boosting
- Advanced time-series models
- Hyperparameter tuning
- Cross-validation designed for time-series data
- Additional calendar and seasonal features

**Explainability**
- SHAP-based model explanations
- Feature-level forecast explanations
- More robust evidence scoring

**Root-Cause Analysis**
- Statistical significance testing
- Causal inference
- Better reference-period selection
- Automated confidence scoring
- More robust product and seller attribution

**NLP**
- Better Portuguese NLP processing
- Sentiment analysis
- Topic modeling
- Transformer-based review analysis

**Production**
- Real-time data ingestion
- Automated anomaly alerts
- Model retraining
- REST API
- Cloud deployment
- Real-time streaming pipelines
- Role-based dashboard access
- Email/Slack notifications

---

## 🏗️ Production Architecture Idea

The current project can be extended toward a production architecture such as:

```text
                Live E-commerce Data
                         │
                         ▼
                  Data Ingestion
                         │
                         ▼
                 Data Validation
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
              Root-Cause Investigation
                         │
                         ▼
                 Evidence Ranking
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
          Alert System          Dashboard/API
```

The current implementation focuses on developing and validating the core forecasting and investigation workflow using historical data.

---

## 💡 Why This Project?

Traditional forecasting systems may answer:

> "How many orders should we expect?"

Traditional anomaly detection may answer:

> "Something unusual happened."

This project combines both ideas with business investigation. The intended workflow is:

```text
What should happen?
        ↓
What actually happened?
        ↓
Was the difference unusual?
        ↓
Which business dimensions changed?
        ↓
What evidence supports each explanation?
        ↓
What are the most likely contributing factors?
```

This makes the project more than a simple machine-learning prediction model. It demonstrates an end-to-end approach combining:

- Machine Learning
- Feature Engineering
- Forecasting
- Anomaly Detection
- Data Analysis
- Business Investigation
- Explainability
- Interactive Visualization

---

## 👤 Author

**Sujal Singh Jhala**
B.Tech Electronics Engineering, Madhav Institute of Technology & Science, Gwalior

**Interests:** Machine Learning · Artificial Intelligence · Data Science · Natural Language Processing · MLOps · Software Engineering

---

## 📄 Disclaimer

This project is an analytical prototype built using a public historical dataset.

The forecasting model provides estimates of expected demand, while the investigation system identifies business signals associated with unusual observations.

Root-cause findings should be interpreted as **evidence-based hypotheses and likely contributing factors**, not guaranteed causal explanations.

The current architecture is designed as a foundation that could eventually be extended to live business data, automated monitoring, production APIs, and real-time operational analytics.

---

### 📝 Note

The previous version of this README referenced MLflow experiment tracking and model-saving as part of the working pipeline. This has been removed since only completed and verified functionality is documented here.

After replacing `README.md`, commit the update:

```bash
git add README.md
git commit -m "Update project README"
git push origin main
```

Then confirm a clean working tree:

```bash
git status
```