import pandas as pd


DATA_PATH = "data/raw"


def load_orders():
    path = f"{DATA_PATH}/olist_orders_dataset.csv"

    orders = pd.read_csv(path)

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"]
    )

    orders["date"] = orders["order_purchase_timestamp"].dt.date

    return orders


def load_order_items():
    path = f"{DATA_PATH}/olist_order_items_dataset.csv"
    return pd.read_csv(path)


def load_products():
    path = f"{DATA_PATH}/olist_products_dataset.csv"
    return pd.read_csv(path)


def load_customers():
    path = f"{DATA_PATH}/olist_customers_dataset.csv"
    return pd.read_csv(path)


def load_payments():
    path = f"{DATA_PATH}/olist_order_payments_dataset.csv"
    return pd.read_csv(path)


def load_reviews():
    path = f"{DATA_PATH}/olist_order_reviews_dataset.csv"
    return pd.read_csv(path)


def load_sellers():
    path = f"{DATA_PATH}/olist_sellers_dataset.csv"
    return pd.read_csv(path)