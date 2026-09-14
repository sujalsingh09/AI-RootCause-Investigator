import pandas as pd


def prepare_product_data(order_items, products):
    data = order_items.merge(
        products[
            [
                "product_id",
                "product_category_name"
            ]
        ],
        on="product_id",
        how="left"
    )

    return data


def get_category_sales(data, orders, target_date):
    target_date = pd.to_datetime(target_date)

    order_dates = orders[
        [
            "order_id",
            "order_purchase_timestamp"
        ]
    ].copy()

    data = data.merge(
        order_dates,
        on="order_id",
        how="left"
    )

    data["order_purchase_timestamp"] = pd.to_datetime(
        data["order_purchase_timestamp"]
    )

    day_data = data[
        data["order_purchase_timestamp"].dt.date
        == target_date.date()
    ]

    category_sales = (
        day_data
        .groupby("product_category_name")
        .size()
        .sort_values(ascending=False)
    )

    return category_sales


def get_category_share(
    data,
    orders,
    target_date,
    category
):
    target_date = pd.to_datetime(target_date)

    order_dates = orders[
        [
            "order_id",
            "order_purchase_timestamp"
        ]
    ].copy()

    data = data.merge(
        order_dates,
        on="order_id",
        how="left"
    )

    data["order_purchase_timestamp"] = pd.to_datetime(
        data["order_purchase_timestamp"]
    )

    day_data = data[
        data["order_purchase_timestamp"].dt.date
        == target_date.date()
    ]

    total_items = len(day_data)

    category_items = len(
        day_data[
            day_data["product_category_name"] == category
        ]
    )

    if total_items == 0:
        return 0

    return category_items / total_items * 100


def get_previous_period_category_share(
    data,
    orders,
    target_date,
    category,
    days=17
):
    target_date = pd.to_datetime(target_date)

    order_dates = orders[
        [
            "order_id",
            "order_purchase_timestamp"
        ]
    ].copy()

    data = data.merge(
        order_dates,
        on="order_id",
        how="left"
    )

    data["order_purchase_timestamp"] = pd.to_datetime(
        data["order_purchase_timestamp"]
    )

    start_date = target_date - pd.Timedelta(days=days)

    previous_data = data[
        (
            data["order_purchase_timestamp"] >= start_date
        )
        &
        (
            data["order_purchase_timestamp"] < target_date
        )
    ]

    total_items = len(previous_data)

    category_items = len(
        previous_data[
            previous_data["product_category_name"] == category
        ]
    )

    if total_items == 0:
        return 0

    return category_items / total_items * 100


def analyze_product_behavior(
    order_items,
    products,
    orders,
    target_date,
    category
):
    target_date = pd.to_datetime(target_date)

    data = prepare_product_data(
        order_items,
        products
    )

    order_dates = orders[
        [
            "order_id",
            "order_purchase_timestamp"
        ]
    ].copy()

    data = data.merge(
        order_dates,
        on="order_id",
        how="left"
    )

    data["order_purchase_timestamp"] = pd.to_datetime(
        data["order_purchase_timestamp"]
    )

    day_data = data[
        (
            data["order_purchase_timestamp"].dt.date
            == target_date.date()
        )
        &
        (
            data["product_category_name"] == category
        )
    ]

    start_date = target_date - pd.Timedelta(days=30)

    previous_data = data[
        (
            data["order_purchase_timestamp"] >= start_date
        )
        &
        (
            data["order_purchase_timestamp"] < target_date
        )
        &
        (
            data["product_category_name"] == category
        )
    ]

    july18_sales = (
        day_data
        .groupby("product_id")
        .size()
        .rename("target_units")
    )

    baseline_sales = (
        previous_data
        .groupby("product_id")
        .size()
        .rename("baseline_30d_units")
    )

    result = pd.concat(
        [
            july18_sales,
            baseline_sales
        ],
        axis=1
    ).fillna(0)

    result["baseline_daily_avg"] = (
        result["baseline_30d_units"] / 30
    )

    first_sale = (
        data
        .groupby("product_id")[
            "order_purchase_timestamp"
        ]
        .min()
        .rename("first_sale_date")
    )

    result = result.join(first_sale)

    result["target_units"] = result["target_units"].astype(int)
    result["baseline_30d_units"] = (
        result["baseline_30d_units"].astype(int)
    )

    def classify(row):
        if row["first_sale_date"].date() == target_date.date():
            return "New Product"

        if row["baseline_30d_units"] == 0:
            return "Reactivated Product"

        if row["target_units"] > row["baseline_daily_avg"]:
            return "Active + Increased"

        return "Active / Normal"

    result["behavior"] = result.apply(
        classify,
        axis=1
    )

    result = result.sort_values(
        "target_units",
        ascending=False
    )

    return result


def summarize_product_behavior(result):

    result = result[
        result["target_units"] > 0
    ].copy()

    if result.empty:
        return pd.DataFrame(
            columns=[
                "behavior",
                "unique_products",
                "total_items",
                "share_percent"
            ]
        )

    summary = (
        result
        .groupby("behavior")
        .agg(
            unique_products=("target_units", "count"),
            total_items=("target_units", "sum")
        )
        .reset_index()
    )

    total_items = summary["total_items"].sum()

    summary["share_percent"] = (
        summary["total_items"] /
        total_items *
        100
    )

    return summary.sort_values(
        "total_items",
        ascending=False
    )
def analyze_seller_contribution(
    order_items,
    products,
    orders,
    target_date,
    category
):
    target_date = pd.to_datetime(target_date)

    data = order_items.merge(
        products[
            [
                "product_id",
                "product_category_name"
            ]
        ],
        on="product_id",
        how="left"
    )

    data = data.merge(
        orders[
            [
                "order_id",
                "order_purchase_timestamp"
            ]
        ],
        on="order_id",
        how="left"
    )

    data["order_purchase_timestamp"] = pd.to_datetime(
        data["order_purchase_timestamp"]
    )

    day_data = data[
        (
            data["order_purchase_timestamp"].dt.date
            == target_date.date()
        )
        &
        (
            data["product_category_name"]
            == category
        )
    ]

    seller_sales = (
        day_data
        .groupby("seller_id")
        .size()
        .sort_values(ascending=False)
    )

    result = seller_sales.reset_index(
        name="items"
    )

    total_items = result["items"].sum()

    if total_items > 0:
        result["share_percent"] = (
            result["items"] /
            total_items *
            100
        )
    else:
        result["share_percent"] = 0

    return result
def analyze_geography(
    orders,
    customers,
    target_date
):
    target_date = pd.to_datetime(target_date)

    data = orders[
        [
            "order_id",
            "customer_id",
            "order_purchase_timestamp"
        ]
    ].merge(
        customers[
            [
                "customer_id",
                "customer_state"
            ]
        ],
        on="customer_id",
        how="left"
    )

    data["order_purchase_timestamp"] = pd.to_datetime(
        data["order_purchase_timestamp"]
    )

    day_data = data[
        data["order_purchase_timestamp"].dt.date
        == target_date.date()
    ]

    state_sales = (
        day_data
        .groupby("customer_state")
        .size()
        .sort_values(ascending=False)
    )

    result = state_sales.reset_index(
        name="orders"
    )

    total_orders = result["orders"].sum()

    if total_orders > 0:
        result["share_percent"] = (
            result["orders"] /
            total_orders *
            100
        )
    else:
        result["share_percent"] = 0

    return result
def analyze_payments(
    payments,
    orders,
    target_date
):
    target_date = pd.to_datetime(target_date)

    data = payments.merge(
        orders[
            [
                "order_id",
                "order_purchase_timestamp"
            ]
        ],
        on="order_id",
        how="left"
    )

    data["order_purchase_timestamp"] = pd.to_datetime(
        data["order_purchase_timestamp"]
    )

    day_data = data[
        data["order_purchase_timestamp"].dt.date
        == target_date.date()
    ]

    payment_summary = (
        day_data
        .groupby("payment_type")
        .size()
        .reset_index(name="transactions")
    )

    total_transactions = payment_summary["transactions"].sum()

    if total_transactions > 0:
        payment_summary["share_percent"] = (
            payment_summary["transactions"] /
            total_transactions *
            100
        )
    else:
        payment_summary["share_percent"] = 0

    return payment_summary.sort_values(
        "transactions",
        ascending=False
    )


def get_payment_metrics(
    payments,
    orders,
    target_date
):
    target_date = pd.to_datetime(target_date)

    data = payments.merge(
        orders[
            [
                "order_id",
                "order_purchase_timestamp"
            ]
        ],
        on="order_id",
        how="left"
    )

    data["order_purchase_timestamp"] = pd.to_datetime(
        data["order_purchase_timestamp"]
    )

    day_data = data[
        data["order_purchase_timestamp"].dt.date
        == target_date.date()
    ]

    metrics = {
        "average_payment_value": day_data[
            "payment_value"
        ].mean(),
        "average_installments": day_data[
            "payment_installments"
        ].mean()
    }

    return metrics
from sklearn.feature_extraction.text import TfidfVectorizer


def analyze_reviews(
    reviews,
    orders,
    target_date
):
    target_date = pd.to_datetime(target_date)

    data = reviews.merge(
        orders[
            [
                "order_id",
                "order_purchase_timestamp"
            ]
        ],
        on="order_id",
        how="left"
    )

    data["order_purchase_timestamp"] = pd.to_datetime(
        data["order_purchase_timestamp"]
    )

    day_data = data[
        data["order_purchase_timestamp"].dt.date
        == target_date.date()
    ].copy()

    day_data["review_score"] = pd.to_numeric(
        day_data["review_score"],
        errors="coerce"
    )

    low_score_rate = (
        day_data["review_score"].isin([1, 2]).mean() * 100
    )

    average_score = day_data[
        "review_score"
    ].mean()

    return {
        "review_count": len(day_data),
        "average_score": average_score,
        "low_score_rate": low_score_rate
    }


def get_review_terms(
    reviews,
    orders,
    target_date,
    max_features=15
):
    target_date = pd.to_datetime(target_date)

    data = reviews.merge(
        orders[
            [
                "order_id",
                "order_purchase_timestamp"
            ]
        ],
        on="order_id",
        how="left"
    )

    data["order_purchase_timestamp"] = pd.to_datetime(
        data["order_purchase_timestamp"]
    )

    day_data = data[
        data["order_purchase_timestamp"].dt.date
        == target_date.date()
    ].copy()

    text = (
        day_data["review_comment_title"]
        .fillna("")
        .astype(str)
        + " "
        +
        day_data["review_comment_message"]
        .fillna("")
        .astype(str)
    )

    text = text[text.str.strip() != ""]

    if len(text) == 0:
        return pd.DataFrame(
            columns=["term", "score"]
        )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2
    )

    matrix = vectorizer.fit_transform(text)

    scores = matrix.mean(axis=0).A1

    terms = pd.DataFrame({
        "term": vectorizer.get_feature_names_out(),
        "score": scores
    })

    return terms.sort_values(
        "score",
        ascending=False
    ).reset_index(drop=True)
def analyze_delivery(
    orders,
    order_items,
    target_date
):
    target_date = pd.to_datetime(target_date)

    data = orders[
        [
            "order_id",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]
    ].copy()

    data["order_purchase_timestamp"] = pd.to_datetime(
        data["order_purchase_timestamp"]
    )

    data["order_delivered_customer_date"] = pd.to_datetime(
        data["order_delivered_customer_date"],
        errors="coerce"
    )

    data["order_estimated_delivery_date"] = pd.to_datetime(
        data["order_estimated_delivery_date"],
        errors="coerce"
    )

    day_data = data[
        data["order_purchase_timestamp"].dt.date
        == target_date.date()
    ].copy()

    day_data["delivery_days"] = (
        day_data["order_delivered_customer_date"]
        - day_data["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    day_data["delivery_difference"] = (
        day_data["order_delivered_customer_date"]
        - day_data["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400

    late_rate = (
        (day_data["delivery_difference"] > 0).mean()
        * 100
    )

    item_data = order_items[
        [
            "order_id",
            "price",
            "freight_value"
        ]
    ].copy()

    day_items = item_data.merge(
        day_data[
            ["order_id"]
        ],
        on="order_id",
        how="inner"
    )

    day_items["freight_price_ratio"] = (
        day_items["freight_value"]
        / day_items["price"].replace(0, pd.NA)
        * 100
    )

    return {
        "average_delivery_days":
            day_data["delivery_days"].mean(),

        "average_delivery_difference":
            day_data["delivery_difference"].mean(),

        "late_delivery_rate":
            late_rate,

        "average_freight":
            day_items["freight_value"].mean(),

        "average_price":
            day_items["price"].mean(),

        "freight_price_ratio":
            day_items["freight_price_ratio"].mean()
    }