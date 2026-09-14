import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def create_daily_orders(orders):
    daily_orders = orders.groupby("date").size()

    daily_orders.index = pd.to_datetime(daily_orders.index)

    daily_orders = daily_orders.asfreq("D", fill_value=0)

    return daily_orders


def create_features(daily_orders):
    data = daily_orders.to_frame(name="orders")

    data["lag_1"] = data["orders"].shift(1)
    data["lag_7"] = data["orders"].shift(7)

    data["rolling_7"] = (
        data["orders"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    data["day_of_week"] = data.index.dayofweek

    data = data.dropna()

    return data


def split_data(data, cutoff_date="2018-07-01"):
    train_data = data[data.index < cutoff_date]
    test_data = data[data.index >= cutoff_date]

    return train_data, test_data


def train_model(train_data):
    features = [
        "lag_1",
        "lag_7",
        "rolling_7",
        "day_of_week"
    ]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train_data[features],
        train_data["orders"]
    )

    return model, features


def evaluate_model(model, test_data, features):
    predictions = model.predict(test_data[features])

    actual = test_data["orders"]

    mae = mean_absolute_error(actual, predictions)

    rmse = mean_squared_error(
        actual,
        predictions
    ) ** 0.5

    r2 = r2_score(actual, predictions)

    results = pd.DataFrame({
        "date": test_data.index,
        "actual": actual.values,
        "predicted": predictions
    })

    results["error"] = (
        results["actual"] -
        results["predicted"]
    )

    return results, mae, rmse, r2


def get_feature_importance(model, features):
    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    })

    return importance.sort_values(
        "importance",
        ascending=False
    )