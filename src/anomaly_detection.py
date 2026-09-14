import pandas as pd


def detect_coverage_issues(data):
    data = data.copy()

    recent_avg = (
        data["actual"]
        .rolling(30, min_periods=7)
        .mean()
    )

    short_avg = (
        data["actual"]
        .rolling(7, min_periods=7)
        .mean()
    )

    data["coverage_issue"] = (
        (short_avg < recent_avg * 0.20)
        & (data["actual"] < recent_avg * 0.20)
    )

    return data


def calculate_anomaly_score(results):
    data = results.copy()

    data["absolute_error"] = (
        data["error"].abs()
    )

    data = detect_coverage_issues(
        data
    )

    valid_errors = data.loc[
        ~data["coverage_issue"],
        "absolute_error"
    ]

    threshold = valid_errors.quantile(
        0.95
    )

    data["anomaly_score"] = (
        data["absolute_error"] / threshold
    )

    data["is_anomaly"] = (
        (data["absolute_error"] >= threshold)
        & (~data["coverage_issue"])
    )

    return data


def get_top_anomalies(results, n=10):
    anomalies = results[
        results["is_anomaly"]
    ].copy()

    anomalies = anomalies.sort_values(
        "anomaly_score",
        ascending=False
    )

    return anomalies.head(n)


def get_anomaly_for_date(results, date):
    date = pd.to_datetime(date)

    result = results[
        results["date"] == date
    ]

    if result.empty:
        return None

    return result.iloc[0]