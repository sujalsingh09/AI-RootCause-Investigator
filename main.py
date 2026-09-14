import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

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
    analyze_delivery
)

from src.investigation import create_investigation_report


MODEL_PATH = "artifacts/models/demand_forecasting_model.pkl"
FEATURE_PATH = "artifacts/feature_importance.csv"

TARGET_DATE = "2018-07-18"
CUTOFF_DATE = "2018-07-01"


def save_model(model):
    os.makedirs(
        "artifacts/models",
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )


def main():
    print("\nAI Root Cause Investigator")
    print("==========================")

    print("\nLoading data...")

    orders = load_orders()
    order_items = load_order_items()
    products = load_products()
    customers = load_customers()
    payments = load_payments()
    reviews = load_reviews()

    print(f"Orders: {len(orders)}")
    print(f"Order items: {len(order_items)}")
    print(f"Products: {len(products)}")
    print(f"Customers: {len(customers)}")
    print(f"Payments: {len(payments)}")
    print(f"Reviews: {len(reviews)}")

    print("\nTraining forecasting model...")

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

    save_model(model)

    importance = get_feature_importance(
        model,
        features
    )

    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    importance.to_csv(
        FEATURE_PATH,
        index=False
    )

    print("\nModel evaluation")
    print("----------------")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.3f}")

    print("\nDetecting anomalies...")

    anomaly_results = calculate_anomaly_score(
        results
    )

    anomaly_result = anomaly_results[
        anomaly_results["date"]
        == pd.to_datetime(TARGET_DATE)
    ].iloc[0]

    print(
        f"Selected date: {TARGET_DATE}"
    )

    print(
        f"Actual orders: "
        f"{int(anomaly_result['actual'])}"
    )

    print(
        f"Expected orders: "
        f"{anomaly_result['predicted']:.2f}"
    )

    print(
        f"Anomaly score: "
        f"{anomaly_result['anomaly_score']:.2f}"
    )

    print(
        f"Anomaly detected: "
        f"{anomaly_result['is_anomaly']}"
    )

    print("\nInvestigating root cause...")

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
        TARGET_DATE
    )

    category = category_sales.index[0]

    day_share = get_category_share(
        product_data,
        orders,
        TARGET_DATE,
        category
    )

    previous_share = (
        get_previous_period_category_share(
            product_data,
            orders,
            TARGET_DATE,
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
        TARGET_DATE,
        category
    )

    product_summary = summarize_product_behavior(
        product_behavior
    )

    seller_analysis = analyze_seller_contribution(
        order_items,
        products,
        orders,
        TARGET_DATE,
        category
    )

    geography = analyze_geography(
        orders,
        customers,
        TARGET_DATE
    )

    payment_analysis = analyze_payments(
        payments,
        orders,
        TARGET_DATE
    )

    review_analysis = analyze_reviews(
        reviews,
        orders,
        TARGET_DATE
    )

    delivery = analyze_delivery(
        orders,
        order_items,
        TARGET_DATE
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

    print("\nRoot Cause Investigation")
    print("========================")

    print(
        f"Date: {report['date'].date()}"
    )

    print(
        f"Actual orders: "
        f"{report['actual_orders']}"
    )

    print(
        f"Expected orders: "
        f"{report['expected_orders']}"
    )

    print(
        f"Demand lift: "
        f"{report['demand_lift_percent']:.2f}%"
    )

    print(
        f"Anomaly: "
        f"{report['is_anomaly']}"
    )

    print(
        f"Primary category: "
        f"{report['primary_category']}"
    )

    print(
        f"Primary cause: "
        f"{report['primary_cause']}"
    )

    print("\nEvidence ranking")
    print("================")

    print(
        report["evidence"].to_string(
            index=False
        )
    )

    print("\nProduct behavior")
    print("================")

    print(
        product_summary.to_string(
            index=False
        )
    )

    print("\nTop sellers")
    print("===========")

    print(
        seller_analysis
        .head(5)
        .to_string(index=False)
    )

    print("\nTop states")
    print("==========")

    print(
        geography
        .head(5)
        .to_string(index=False)
    )

    print("\nPayment behavior")
    print("================")

    print(
        payment_analysis.to_string(
            index=False
        )
    )

    print("\nReview analysis")
    print("===============")

    print(
        f"Average score: "
        f"{review_analysis['average_score']:.2f}"
    )

    print(
        f"Low score rate: "
        f"{review_analysis['low_score_rate']:.2f}%"
    )

    print("\nDelivery analysis")
    print("=================")

    print(
        f"Average delivery days: "
        f"{delivery['average_delivery_days']:.2f}"
    )

    print(
        f"Late delivery rate: "
        f"{delivery['late_delivery_rate']:.2f}%"
    )

    print("\nTracking experiment with MLflow...")

    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    mlflow.set_experiment(
        "ecommerce-demand-forecasting"
    )

    with mlflow.start_run():

        mlflow.log_params({
            "model": "RandomForestRegressor",
            "n_estimators": 200,
            "random_state": 42,
            "cutoff_date": CUTOFF_DATE,
            "target_date": TARGET_DATE
        })

        mlflow.log_metrics({
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "demand_lift": report[
                "demand_lift_percent"
            ],
            "anomaly_score": report[
                "anomaly_score"
            ]
        })

        mlflow.log_artifact(
            MODEL_PATH
        )

        mlflow.log_artifact(
            FEATURE_PATH
        )

        mlflow.sklearn.log_model(
            model,
            name="demand_model"
        )

    print("\nPipeline completed successfully.")
    print(
        f"Model saved: {MODEL_PATH}"
    )
    print(
        f"Feature importance saved: {FEATURE_PATH}"
    )


if __name__ == "__main__":
    main()