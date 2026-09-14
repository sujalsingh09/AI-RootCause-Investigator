import pandas as pd


def calculate_demand_lift(actual, expected):
    if expected == 0:
        return 0

    return ((actual - expected) / expected) * 100


def get_product_share(product_summary, behavior):
    total = behavior["target_units"].sum()

    if total == 0:
        return 0

    return (
        behavior["target_units"].sum()
        / total
    ) * 100


def build_evidence(
    category_share_change,
    product_summary,
    seller_analysis=None,
    geography=None,
    payment_analysis=None,
    review_analysis=None,
    delivery=None
):

    evidence = []

    category_score = min(
        10,
        5 + abs(category_share_change)
    )

    evidence.append({
        "factor": "Category demand surge",
        "score": round(category_score, 2),
        "status": "Primary",
        "detail": (
            f"Category share changed by "
            f"{category_share_change:+.2f} percentage points"
        )
    })

    for _, row in product_summary.iterrows():

        if row["total_items"] <= 0:
            continue

        behavior = row["behavior"]
        share = row["share_percent"]

        if behavior == "Reactivated Product":
            score = min(10, 5 + share / 10)
            status = "Supporting"

        elif behavior == "Active + Increased":
            score = min(10, 5 + share / 10)
            status = "Supporting"

        elif behavior == "New Product":
            score = min(10, 5 + share / 10)
            status = "Supporting"

        else:
            continue

        evidence.append({
            "factor": behavior + "s",
            "score": round(score, 2),
            "status": status,
            "detail": (
                f"{behavior}s contributed "
                f"{share:.2f}% of category items"
            )
        })

    if seller_analysis is not None:

        top_seller_share = (
            seller_analysis.iloc[0]["share_percent"]
        )

        if top_seller_share >= 30:
            seller_score = 7
        elif top_seller_share >= 20:
            seller_score = 5
        else:
            seller_score = 3

        evidence.append({
            "factor": "Seller concentration",
            "score": seller_score,
            "status": "Weak",
            "detail": (
                f"Top seller contributed "
                f"{top_seller_share:.2f}% of category items"
            )
        })

    if geography is not None:

        top_state_share = (
            geography.iloc[0]["share_percent"]
        )

        evidence.append({
            "factor": "Geographic concentration",
            "score": 3,
            "status": "Weak",
            "detail": (
                f"Top state contributed "
                f"{top_state_share:.2f}% of anomaly-day orders; "
                f"high share alone does not establish causation"
            )
        })

    if payment_analysis is not None:

        top_payment_share = (
            payment_analysis.iloc[0]["share_percent"]
        )

        evidence.append({
            "factor": "Payment behavior",
            "score": 2,
            "status": "Weak",
            "detail": (
                f"Top payment method represented "
                f"{top_payment_share:.2f}% of transactions; "
                f"share alone does not establish causation"
            )
        })

    if review_analysis is not None:

        low_score_rate = (
            review_analysis["low_score_rate"]
        )

        evidence.append({
            "factor": "Customer dissatisfaction",
            "score": 2,
            "status": "Context",
            "detail": (
                f"Low review score rate was "
                f"{low_score_rate:.2f}%"
            )
        })

    if delivery is not None:

        late_rate = (
            delivery["late_delivery_rate"]
        )

        evidence.append({
            "factor": "Delivery problems",
            "score": 2,
            "status": "Context",
            "detail": (
                f"Late delivery rate was "
                f"{late_rate:.2f}%"
            )
        })

    return pd.DataFrame(evidence)


def rank_evidence(evidence):
    return evidence.sort_values(
        "score",
        ascending=False
    ).reset_index(drop=True)


def create_investigation_report(
    anomaly_result,
    category,
    category_share_change,
    product_summary,
    seller_analysis=None,
    geography=None,
    payment_analysis=None,
    review_analysis=None,
    delivery=None
):

    actual = anomaly_result["actual"]
    expected = anomaly_result["predicted"]

    demand_lift = calculate_demand_lift(
        actual,
        expected
    )

    evidence = build_evidence(
        category_share_change,
        product_summary,
        seller_analysis,
        geography,
        payment_analysis,
        review_analysis,
        delivery
    )

    evidence = rank_evidence(
        evidence
    )

    return {
        "date": anomaly_result["date"],
        "actual_orders": int(actual),
        "expected_orders": float(expected),
        "demand_lift_percent": demand_lift,
        "anomaly_score": float(
            anomaly_result["anomaly_score"]
        ),
        "is_anomaly": bool(
            anomaly_result["is_anomaly"]
        ),
        "primary_category": category,
        "primary_cause": "Category demand surge",
        "category_share_change": category_share_change,
        "evidence": evidence
    }