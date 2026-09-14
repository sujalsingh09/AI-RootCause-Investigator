import sys
import os

import pandas as pd
import streamlit as st

sys.path.append(
    os.path.dirname(os.path.dirname(__file__))
)

from src.data_loader import (
    load_orders,
    load_order_items,
    load_products,
    load_customers,
    load_payments,
    load_reviews
)

from src.forecasting import (
    create_daily_orders,
    create_features,
    split_data,
    train_model,
    evaluate_model,
    get_feature_importance
)

from src.anomaly_detection import calculate_anomaly_score

from src.product_analysis import (
    get_category_sales,
    get_category_share,
    get_previous_period_category_share,
    analyze_product_behavior,
    summarize_product_behavior,
    analyze_seller_contribution,
    analyze_geography,
    analyze_payments,
    analyze_reviews,
    analyze_delivery,
    get_review_terms
)

from src.investigation import create_investigation_report


CUTOFF_DATE = "2018-07-01"


st.set_page_config(
    page_title="AI Root Cause Investigator",
    page_icon="🔎",
    layout="wide"
)


@st.cache_data
def load_data():

    orders = load_orders()
    order_items = load_order_items()
    products = load_products()
    customers = load_customers()
    payments = load_payments()
    reviews = load_reviews()

    return (
        orders,
        order_items,
        products,
        customers,
        payments,
        reviews
    )


@st.cache_resource
def run_model(orders):

    daily_orders = create_daily_orders(
        orders
    )

    data = create_features(
        daily_orders
    )

    train_data, test_data = split_data(
        data,
        CUTOFF_DATE
    )

    model, features = train_model(
        train_data
    )

    results, mae, rmse, r2 = evaluate_model(
        model,
        test_data,
        features
    )

    importance = get_feature_importance(
        model,
        features
    )

    return (
        model,
        results,
        importance,
        mae,
        rmse,
        r2
    )


@st.cache_data
def prepare_investigation(
    target_date,
    orders,
    order_items,
    products,
    customers,
    payments,
    reviews,
    results
):

    anomaly_results = calculate_anomaly_score(
        results
    )

    target_date = pd.to_datetime(
        target_date
    )

    selected = anomaly_results[
        anomaly_results["date"] == target_date
    ]

    if selected.empty:
        return None

    anomaly_result = selected.iloc[0]

    product_data = order_items.merge(
        products[
            [
                "product_id",
                "product_category_name"
            ]
        ],
        on="product_id",
        how="left"
    )

    category_sales = get_category_sales(
        product_data,
        orders,
        target_date
    )

    if category_sales.empty:
        return None

    category = category_sales.index[0]

    day_share = get_category_share(
        product_data,
        orders,
        target_date,
        category
    )

    previous_share = (
        get_previous_period_category_share(
            product_data,
            orders,
            target_date,
            category
        )
    )

    category_share_change = (
        day_share - previous_share
    )

    product_behavior = analyze_product_behavior(
        order_items,
        products,
        orders,
        target_date,
        category
    )

    product_summary = summarize_product_behavior(
        product_behavior
    )

    product_summary = product_summary[
        product_summary["total_items"] > 0
    ].copy()

    seller_analysis = analyze_seller_contribution(
        order_items,
        products,
        orders,
        target_date,
        category
    )

    geography = analyze_geography(
        orders,
        customers,
        target_date
    )

    payment_analysis = analyze_payments(
        payments,
        orders,
        target_date
    )

    review_analysis = analyze_reviews(
        reviews,
        orders,
        target_date
    )
    review_terms = get_review_terms(
    reviews,
    orders,
    target_date
    )

    delivery = analyze_delivery(
        orders,
        order_items,
        target_date
    )

    report = create_investigation_report(
        anomaly_result=anomaly_result,
        category=category,
        category_share_change=category_share_change,
        product_summary=product_summary,
        seller_analysis=seller_analysis,
        geography=geography,
        payment_analysis=payment_analysis,
        review_analysis=review_analysis,
        delivery=delivery
    )

    return {
        "report": report,
        "anomaly_results": anomaly_results,
        "category_sales": category_sales,
        "category_share_change": category_share_change,
        "product_summary": product_summary,
        "seller_analysis": seller_analysis,
        "geography": geography,
        "payment_analysis": payment_analysis,
        "review_analysis": review_analysis,
        "review_terms": review_terms,
        "delivery": delivery
    }


with st.spinner("Loading e-commerce data..."):

    (
        orders,
        order_items,
        products,
        customers,
        payments,
        reviews
    ) = load_data()


with st.spinner("Training forecasting model..."):

    (
        model,
        results,
        importance,
        mae,
        rmse,
        r2
    ) = run_model(orders)
anomaly_results = calculate_anomaly_score(
    results
)

anomaly_list = anomaly_results[
    anomaly_results["is_anomaly"] == True
].copy()

anomaly_list = anomaly_list.sort_values(
    "anomaly_score",
    ascending=False
)


# Only dates available in the model holdout period
available_dates = (
    results["date"]
    .sort_values()
    .dt.date
    .tolist()
)


st.title("🔎 AI Root Cause Investigator")

st.write(
    "Forecast demand, detect unusual activity and investigate "
    "possible business root causes."
)


st.sidebar.markdown(
    "### 🚨 Top Anomalies"
)

top_anomalies = anomaly_list.head(10)[
    [
        "date",
        "actual",
        "predicted",
        "anomaly_score"
    ]
].copy()

top_anomalies["date"] = (
    top_anomalies["date"]
    .dt.strftime("%Y-%m-%d")
)

top_anomalies = top_anomalies.rename(
    columns={
        "date": "Date",
        "actual": "Actual",
        "predicted": "Expected",
        "anomaly_score": "Score"
    }
)

st.sidebar.dataframe(
    top_anomalies,
    use_container_width=True,
    hide_index=True
)

st.sidebar.markdown(
    "### 📅 Investigation Date"
)

min_date = min(available_dates)
max_date = max(available_dates)

selected_date = st.sidebar.date_input(
    "Select a test date",
    value=pd.to_datetime("2018-07-18").date(),
    min_value=min_date,
    max_value=max_date
)

if selected_date not in available_dates:

    st.sidebar.warning(
        "This date is outside the forecasting test period."
    )

else:

    st.sidebar.caption(
        "Select any date from the forecasting test period."
    )

investigation = None

if selected_date not in available_dates:

    st.warning(
        f"{selected_date} is outside the forecasting test period."
    )

    st.info(
        f"Please select a date between {min_date} and {max_date}."
    )

    st.stop()


investigation = prepare_investigation(
    selected_date,
    orders,
    order_items,
    products,
    customers,
    payments,
    reviews,
    results
)


if investigation is None:

    st.warning(
        f"Detailed investigation is not available for "
        f"{selected_date}."
    )

    st.info(
        "The forecasting model has data for this date, but "
        "the detailed product and category data required for "
        "root-cause investigation is insufficient. "
        "This is a data coverage limitation, not a model error."
    )

    st.stop()

report = investigation["report"]

anomaly_results = investigation[
    "anomaly_results"
]

category_sales = investigation[
    "category_sales"
]

category_share_change = investigation[
    "category_share_change"
]

product_summary = investigation[
    "product_summary"
]

seller_analysis = investigation[
    "seller_analysis"
]

geography = investigation[
    "geography"
]

payment_analysis = investigation[
    "payment_analysis"
]

review_analysis = investigation[
    "review_analysis"
]
review_terms = investigation[
    "review_terms"
]

delivery = investigation[
    "delivery"
]


# --------------------------------------------------
# Summary
# --------------------------------------------------

st.divider()

st.subheader(
    f"🚨 Investigation: {selected_date}"
)


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Actual Orders",
        report["actual_orders"]
    )

with col2:

    st.metric(
        "Expected Orders",
        f"{report['expected_orders']:.0f}"
    )

with col3:

    st.metric(
        "Demand Lift",
        f"{report['demand_lift_percent']:.2f}%"
    )

with col4:

    st.metric(
        "Anomaly Score",
        f"{report['anomaly_score']:.2f}"
    )

if report["is_anomaly"]:

    lift = report["demand_lift_percent"]

    if lift >= 50:
        severity = "High"
    elif lift >= 25:
        severity = "Medium"
    else:
        severity = "Low"

    st.warning(
        f"Anomaly detected: {report['actual_orders']} orders "
        f"were recorded against an expected "
        f"{report['expected_orders']:.0f} orders."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Anomaly Severity",
            severity
        )

    with col2:
        st.metric(
            "Demand Deviation",
            f"{lift:+.2f}%"
        )

else:

    severity = "Normal"

    st.success(
        "No significant demand anomaly was detected for this date."
    )


# --------------------------------------------------
# Root cause
# --------------------------------------------------

st.subheader("🎯 Root Cause")

col1, col2 = st.columns(2)

with col1:

    st.markdown("**Primary Category**")

    st.info(
        report["primary_category"]
    )

with col2:

    st.markdown("**Primary Cause**")

    if report["is_anomaly"]:

        st.info(
            report["primary_cause"]
        )

    else:

        st.info(
            "No significant root cause identified"
        )


if report["is_anomaly"]:

    st.write(
        "The strongest signal identified by the investigation "
        "comes from changes in category-level demand."
    )

else:

    st.write(
        "No significant demand anomaly was detected, so "
        "no primary root cause was assigned."
    )
    
st.markdown("### 💡 Business Recommendation")

if report["is_anomaly"]:

    st.write(
        f"Prioritize investigation of the "
        f"{report['primary_category']} category. "
        f"Review product availability, seller activity and "
        f"recent demand changes in this category before making "
        f"inventory or operational decisions."
    )

else:

    st.write(
        "No immediate anomaly-driven action is required."
    )

# --------------------------------------------------
# Forecast
# --------------------------------------------------

st.subheader("📈 Demand Forecast vs Actual")

forecast_chart = anomaly_results[
    [
        "date",
        "actual",
        "predicted"
    ]
].copy()

forecast_chart = forecast_chart.set_index(
    "date"
)

forecast_chart = forecast_chart.rename(
    columns={
        "actual": "Actual",
        "predicted": "Expected"
    }
)

selected_timestamp = pd.to_datetime(
    selected_date
)

start_date = (
    selected_timestamp
    - pd.Timedelta(days=14)
)

end_date = (
    selected_timestamp
    + pd.Timedelta(days=14)
)

forecast_chart = forecast_chart[
    (
        forecast_chart.index >= start_date
    )
    &
    (
        forecast_chart.index <= end_date
    )
]

st.line_chart(
    forecast_chart,
    use_container_width=True
)


# --------------------------------------------------
# Evidence
# --------------------------------------------------

st.subheader("🔍 Evidence Ranking")

evidence = report["evidence"].copy()

evidence = evidence.rename(
    columns={
        "factor": "Factor",
        "score": "Evidence Score",
        "status": "Status",
        "detail": "Explanation"
    }
)

st.dataframe(
    evidence,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Evidence scores are heuristic signals used to rank possible "
    "root causes. They do not prove causation."
)


# --------------------------------------------------
# Category
# --------------------------------------------------

st.subheader("📦 Category Investigation")

col1, col2 = st.columns(2)

with col1:

    category_chart = (
        category_sales
        .head(10)
        .sort_values(ascending=True)
    )

    st.bar_chart(
        category_chart,
        use_container_width=True
    )

with col2:

    st.metric(
        "Primary Category",
        report["primary_category"]
    )

    st.metric(
        "Category Share Change",
        f"{category_share_change:+.2f} pp"
    )


# --------------------------------------------------
# Product
# --------------------------------------------------

st.subheader("🧩 Product Behavior")

product_chart = product_summary[
    [
        "behavior",
        "share_percent"
    ]
].copy()

product_chart = product_chart.sort_values(
    "share_percent",
    ascending=True
)

product_chart = product_chart.set_index(
    "behavior"
)

st.bar_chart(
    product_chart,
    use_container_width=True
)
st.dataframe(
    product_summary,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# Sellers and geography
# --------------------------------------------------

st.subheader(
    "🏪 Seller & Geographic Investigation"
)

col1, col2 = st.columns(2)

with col1:

    st.markdown("### Top Sellers")

    st.dataframe(
        seller_analysis.head(10),
        use_container_width=True,
        hide_index=True
    )

with col2:

    st.markdown("### Top States")

    state_chart = geography[
        [
            "customer_state",
            "orders"
        ]
    ].head(10)

    state_chart = state_chart.set_index(
        "customer_state"
    )

    st.bar_chart(
        state_chart,
        use_container_width=True
    )


# --------------------------------------------------
# Payments
# --------------------------------------------------

st.subheader("💳 Payment Behavior")

col1, col2 = st.columns(2)

with col1:

    payment_chart = payment_analysis[
        [
            "payment_type",
            "share_percent"
        ]
    ].copy()

    payment_chart = payment_chart.set_index(
        "payment_type"
    )

    st.bar_chart(
        payment_chart,
        use_container_width=True
    )

with col2:

    st.dataframe(
        payment_analysis,
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# Customer experience
# --------------------------------------------------

st.subheader("⭐ Customer Experience")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Average Review Score",
        f"{review_analysis['average_score']:.2f} / 5"
    )

with col2:

    st.metric(
        "Low Score Rate",
        f"{review_analysis['low_score_rate']:.2f}%"
    )


# --------------------------------------------------
# Delivery
# --------------------------------------------------
st.subheader("🧠 Review Text Analysis")

if review_terms is not None and not review_terms.empty:

    st.write(
        "Important terms found in customer review text."
    )

    st.dataframe(
        review_terms,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No review text was available for this date."
    )
st.subheader("🚚 Delivery Performance")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Average Delivery Time",
        f"{delivery['average_delivery_days']:.2f} days"
    )

with col2:

    st.metric(
        "Late Delivery Rate",
        f"{delivery['late_delivery_rate']:.2f}%"
    )


# --------------------------------------------------
# Model
# --------------------------------------------------

st.subheader("🤖 Forecasting Model")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "MAE",
        f"{mae:.2f}"
    )

with col2:

    st.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

with col3:

    st.metric(
        "R²",
        f"{r2:.3f}"
    )


st.markdown("### Feature Importance")

importance_chart = importance[
    [
        "feature",
        "importance"
    ]
].copy()

importance_chart = importance_chart.set_index(
    "feature"
)

st.bar_chart(
    importance_chart,
    use_container_width=True
)

# --------------------------------------------------
# Conclusion
# --------------------------------------------------

st.divider()

st.subheader("📝 Investigation Conclusion")

if report["is_anomaly"]:

    st.write(
        f"On {selected_date}, the system detected a "
        f"{report['demand_lift_percent']:.2f}% demand deviation "
        f"from the model's expected order volume."
    )

    st.write(
        f"The investigation identified "
        f"**{report['primary_category']}** as the strongest "
        f"category-level signal. Its share of orders changed by "
        f"**{category_share_change:+.2f} percentage points**."
    )

    st.write(
        "Product-level analysis showed that the category increase "
        "was distributed across reactivated products, existing "
        "products with increased activity and new products rather "
        "than being driven by a single product."
    )

    st.write(
        "Seller, payment, customer review and delivery signals "
        "did not provide stronger evidence than the category-level "
        "demand signal."
    )

    st.info(
        "Investigation result: the most likely explanation is a "
        "broad category demand surge. This is an observational "
        "finding from historical data and should not be interpreted "
        "as proof of causation."
    )

else:

    st.write(
        f"No major demand anomaly was detected on {selected_date}."
    )
st.caption(
    "AI Root Cause Investigator | Historical prototype using "
    "the Olist Brazilian e-commerce dataset"
)