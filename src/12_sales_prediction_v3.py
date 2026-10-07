from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.model_selection import (
    TimeSeriesSplit,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURACION
# ============================================================

DATA_FILE = Path("outputs/online_retail_clean.csv")

OUTPUT_DIR = Path("outputs/prediction_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_START = pd.Timestamp("2011-07-20")
RANDOM_STATE = 42

print("=" * 82)
print("PREDICCION DE VENTAS V3")
print("=" * 82)


# ============================================================
# 1. CARGAR DATOS
# ============================================================

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["InvoiceDate"],
)

df["Date"] = (
    df["InvoiceDate"]
    .dt.normalize()
)


# ============================================================
# 2. SERIE DIARIA
# ============================================================

daily = (
    df.groupby("Date")
    .agg(
        Sales=("Sales", "sum"),
        Quantity=("Quantity", "sum"),
        Invoices=("Invoice", "nunique"),
        Customers=("CustomerID", "nunique"),
    )
    .sort_index()
)

calendar = pd.date_range(
    daily.index.min(),
    daily.index.max(),
    freq="D",
)

daily = daily.reindex(
    calendar,
    fill_value=0,
)

daily.index.name = "Date"

print(
    f"\nDias calendario: {len(daily):,}"
)

print(
    f"Dias con ventas: "
    f"{(daily['Sales'] > 0).sum():,}"
)


# ============================================================
# 3. CALENDARIO
# ============================================================

daily["DayOfWeek"] = (
    daily.index.dayofweek
)

daily["Month"] = (
    daily.index.month
)

daily["DayOfMonth"] = (
    daily.index.day
)

daily["DayOfYear"] = (
    daily.index.dayofyear
)

daily["WeekOfYear"] = (
    daily.index
    .isocalendar()
    .week
    .astype(int)
)

daily["Quarter"] = (
    daily.index.quarter
)

daily["IsWeekend"] = (
    daily["DayOfWeek"] >= 5
).astype(int)

daily["IsSaturday"] = (
    daily["DayOfWeek"] == 5
).astype(int)

daily["IsSunday"] = (
    daily["DayOfWeek"] == 6
).astype(int)

daily["IsMonthStart"] = (
    daily.index.is_month_start
).astype(int)

daily["IsMonthEnd"] = (
    daily.index.is_month_end
).astype(int)

daily["PeakSeason"] = (
    daily["Month"]
    .isin([9, 10, 11])
).astype(int)

daily["IsDecember"] = (
    daily["Month"] == 12
).astype(int)


# ============================================================
# 4. VARIABLES CICLICAS
# ============================================================

daily["DowSin"] = np.sin(
    2 * np.pi *
    daily["DayOfWeek"] / 7
)

daily["DowCos"] = np.cos(
    2 * np.pi *
    daily["DayOfWeek"] / 7
)

daily["YearSin"] = np.sin(
    2 * np.pi *
    daily["DayOfYear"] / 365.25
)

daily["YearCos"] = np.cos(
    2 * np.pi *
    daily["DayOfYear"] / 365.25
)


# ============================================================
# 5. LAGS DE VENTAS
# ============================================================

sales_lags = [
    1,
    2,
    3,
    6,
    7,
    8,
    14,
    21,
    28,
    35,
]

for lag in sales_lags:

    daily[
        f"SalesLag{lag}"
    ] = daily[
        "Sales"
    ].shift(lag)


# ============================================================
# 6. ESTADISTICAS MOVILES
# ============================================================

past_sales = (
    daily["Sales"]
    .shift(1)
)

for window in [
    7,
    14,
    28,
]:

    daily[
        f"RollingMean{window}"
    ] = (
        past_sales
        .rolling(window)
        .mean()
    )

    daily[
        f"RollingMedian{window}"
    ] = (
        past_sales
        .rolling(window)
        .median()
    )

    daily[
        f"RollingStd{window}"
    ] = (
        past_sales
        .rolling(window)
        .std()
    )

daily["RollingMin7"] = (
    past_sales
    .rolling(7)
    .min()
)

daily["RollingMax7"] = (
    past_sales
    .rolling(7)
    .max()
)


# ============================================================
# 7. MEMORIA DEL MISMO DIA DE LA SEMANA
# ============================================================

weekly_lag_columns = [
    "SalesLag7",
    "SalesLag14",
    "SalesLag21",
    "SalesLag28",
]

daily[
    "SameWeekdayMean4"
] = (
    daily[
        weekly_lag_columns
    ]
    .mean(axis=1)
)

daily[
    "SameWeekdayMedian4"
] = (
    daily[
        weekly_lag_columns
    ]
    .median(axis=1)
)

daily[
    "SameWeekdayStd4"
] = (
    daily[
        weekly_lag_columns
    ]
    .std(axis=1)
)

daily[
    "SameWeekdayMin4"
] = (
    daily[
        weekly_lag_columns
    ]
    .min(axis=1)
)

daily[
    "SameWeekdayMax4"
] = (
    daily[
        weekly_lag_columns
    ]
    .max(axis=1)
)


# ============================================================
# 8. TENDENCIA SEMANAL
# ============================================================

daily["WeeklyTrend1"] = (
    daily["SalesLag7"] -
    daily["SalesLag14"]
)

daily["WeeklyTrend2"] = (
    daily["SalesLag14"] -
    daily["SalesLag21"]
)

daily["WeeklyAcceleration"] = (
    daily["WeeklyTrend1"] -
    daily["WeeklyTrend2"]
)

daily["WeeklyRatio"] = (
    daily["SalesLag7"] /
    (
        daily[
            [
                "SalesLag14",
                "SalesLag21",
                "SalesLag28",
            ]
        ]
        .mean(axis=1)
        .abs()
        + 1
    )
)


# ============================================================
# 9. ACTIVIDAD OPERATIVA PASADA
# ============================================================

operational_columns = [
    "Quantity",
    "Invoices",
    "Customers",
]

operational_lags = [
    1,
    7,
    14,
    28,
]

for col in operational_columns:

    for lag in operational_lags:

        daily[
            f"{col}Lag{lag}"
        ] = (
            daily[col]
            .shift(lag)
        )

    past = (
        daily[col]
        .shift(1)
    )

    daily[
        f"{col}RollingMean7"
    ] = (
        past
        .rolling(7)
        .mean()
    )

    daily[
        f"{col}RollingMean28"
    ] = (
        past
        .rolling(28)
        .mean()
    )


# ============================================================
# 10. FEATURES
# ============================================================

calendar_features = [
    "DayOfWeek",
    "Month",
    "DayOfMonth",
    "DayOfYear",
    "WeekOfYear",
    "Quarter",
    "IsWeekend",
    "IsSaturday",
    "IsSunday",
    "IsMonthStart",
    "IsMonthEnd",
    "PeakSeason",
    "IsDecember",
    "DowSin",
    "DowCos",
    "YearSin",
    "YearCos",
]

lag_features = [
    f"SalesLag{x}"
    for x in sales_lags
]

rolling_features = [
    "RollingMean7",
    "RollingMean14",
    "RollingMean28",
    "RollingMedian7",
    "RollingMedian14",
    "RollingMedian28",
    "RollingStd7",
    "RollingStd14",
    "RollingStd28",
    "RollingMin7",
    "RollingMax7",
]

weekly_features = [
    "SameWeekdayMean4",
    "SameWeekdayMedian4",
    "SameWeekdayStd4",
    "SameWeekdayMin4",
    "SameWeekdayMax4",
    "WeeklyTrend1",
    "WeeklyTrend2",
    "WeeklyAcceleration",
    "WeeklyRatio",
]

operational_features = []

for col in operational_columns:

    for lag in operational_lags:

        operational_features.append(
            f"{col}Lag{lag}"
        )

    operational_features.extend(
        [
            f"{col}RollingMean7",
            f"{col}RollingMean28",
        ]
    )


features = (
    calendar_features +
    lag_features +
    rolling_features +
    weekly_features +
    operational_features
)

model_data = (
    daily[
        ["Sales"] + features
    ]
    .dropna()
    .copy()
)

print(
    f"\nObservaciones modelables: "
    f"{len(model_data):,}"
)

print(
    f"Numero de features: "
    f"{len(features)}"
)


# ============================================================
# 11. SPLIT EXACTAMENTE IGUAL A V1/V2
# ============================================================

train = (
    model_data[
        model_data.index <
        TEST_START
    ]
    .copy()
)

test = (
    model_data[
        model_data.index >=
        TEST_START
    ]
    .copy()
)

X_train = train[
    features
]

y_train = train[
    "Sales"
]

X_test = test[
    features
]

y_test = test[
    "Sales"
]

print("\nSplit:")

print(
    f"Train: "
    f"{train.index.min().date()} "
    f"-> "
    f"{train.index.max().date()} "
    f"({len(train)} dias)"
)

print(
    f"Test:  "
    f"{test.index.min().date()} "
    f"-> "
    f"{test.index.max().date()} "
    f"({len(test)} dias)"
)


# ============================================================
# 12. APRENDER CALENDARIO SOLO CON TRAIN
# ============================================================

calendar_train = (
    daily[
        daily.index <
        TEST_START
    ][["Sales"]]
    .copy()
)

calendar_train[
    "DayOfWeekName"
] = (
    calendar_train
    .index
    .day_name()
)

calendar_train[
    "ZeroSales"
] = (
    calendar_train["Sales"] == 0
)

zero_rate = (
    calendar_train
    .groupby(
        "DayOfWeekName"
    )["ZeroSales"]
    .mean()
)

closed_weekdays = (
    zero_rate[
        zero_rate >= 0.95
    ]
    .index
    .tolist()
)

print(
    "\nDias estructuralmente cerrados "
    "aprendidos de TRAIN:"
)

print(
    closed_weekdays
)


# ============================================================
# 13. METRICAS
# ============================================================

def smape(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    denominator = (
        np.abs(y_true) +
        np.abs(y_pred)
    )

    valid = (
        denominator != 0
    )

    return (
        100 *
        np.mean(
            2 *
            np.abs(
                y_pred[valid] -
                y_true[valid]
            )
            /
            denominator[valid]
        )
    )


def evaluate(
    name,
    actual,
    predicted,
):

    predicted = np.maximum(
        np.asarray(predicted),
        0,
    )

    return {
        "Model":
            name,

        "MAE":
            mean_absolute_error(
                actual,
                predicted,
            ),

        "RMSE":
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            ),

        "R2":
            r2_score(
                actual,
                predicted,
            ),

        "sMAPE":
            smape(
                actual,
                predicted,
            ),

        "MedianAE":
            np.median(
                np.abs(
                    np.asarray(actual) -
                    predicted
                )
            ),
    }


def apply_calendar(
    predictions,
    dates,
):

    predictions = (
        np.asarray(
            predictions
        )
        .copy()
    )

    weekdays = (
        pd.Series(
            dates.day_name(),
            index=dates,
        )
    )

    mask = (
        weekdays
        .isin(
            closed_weekdays
        )
        .values
    )

    predictions[
        mask
    ] = 0.0

    return predictions


# ============================================================
# 14. BASELINE
# ============================================================

results = []

baseline_lag7 = (
    X_test[
        "SalesLag7"
    ].values
)

baseline_lag7 = (
    apply_calendar(
        baseline_lag7,
        test.index,
    )
)

results.append(
    evaluate(
        "Baseline Lag-7",
        y_test,
        baseline_lag7,
    )
)


# ============================================================
# 15. TIME SERIES CV
# ============================================================

tscv = TimeSeriesSplit(
    n_splits=5
)


# ============================================================
# 16. EXTRA TREES - SQUARED ERROR
# ============================================================

print("\n" + "=" * 82)
print("EXTRA TREES V3 - SQUARED ERROR")
print("=" * 82)

et_squared = ExtraTreesRegressor(
    random_state=RANDOM_STATE,
    n_jobs=-1,
    criterion="squared_error",
)

params_squared = {
    "n_estimators": [
        300,
        500,
        700,
        900,
    ],
    "max_depth": [
        6,
        8,
        10,
        12,
        16,
        None,
    ],
    "min_samples_split": [
        2,
        4,
        6,
        8,
        12,
    ],
    "min_samples_leaf": [
        1,
        2,
        4,
        6,
        8,
        10,
    ],
    "max_features": [
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        1.0,
        "sqrt",
    ],
}

search_squared = RandomizedSearchCV(
    estimator=et_squared,
    param_distributions=params_squared,
    n_iter=24,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)

search_squared.fit(
    X_train,
    y_train,
)

best_squared = (
    search_squared
    .best_estimator_
)

print(
    "\nMejores parametros:"
)

print(
    search_squared
    .best_params_
)

print(
    f"MAE CV: "
    f"{-search_squared.best_score_:,.2f}"
)


# ============================================================
# 17. EXTRA TREES - ABSOLUTE ERROR
# ============================================================

print("\n" + "=" * 82)
print("EXTRA TREES V3 - ABSOLUTE ERROR")
print("=" * 82)

et_absolute = ExtraTreesRegressor(
    random_state=RANDOM_STATE,
    n_jobs=-1,
    criterion="absolute_error",
)

params_absolute = {
    "n_estimators": [
        300,
        500,
        700,
    ],
    "max_depth": [
        6,
        8,
        10,
        12,
        16,
        None,
    ],
    "min_samples_split": [
        2,
        4,
        6,
        8,
        12,
    ],
    "min_samples_leaf": [
        1,
        2,
        4,
        6,
        8,
        10,
    ],
    "max_features": [
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        1.0,
    ],
}

search_absolute = RandomizedSearchCV(
    estimator=et_absolute,
    param_distributions=params_absolute,
    n_iter=18,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)

search_absolute.fit(
    X_train,
    y_train,
)

best_absolute = (
    search_absolute
    .best_estimator_
)

print(
    "\nMejores parametros:"
)

print(
    search_absolute
    .best_params_
)

print(
    f"MAE CV: "
    f"{-search_absolute.best_score_:,.2f}"
)


# ============================================================
# 18. GRADIENT BOOSTING ROBUSTO
# ============================================================

print("\n" + "=" * 82)
print("GRADIENT BOOSTING HUBER")
print("=" * 82)

gbr = GradientBoostingRegressor(
    random_state=RANDOM_STATE,
    loss="huber",
)

params_gbr = {
    "n_estimators": [
        100,
        200,
        300,
        400,
    ],
    "learning_rate": [
        0.02,
        0.03,
        0.05,
        0.08,
        0.1,
    ],
    "max_depth": [
        2,
        3,
        4,
        5,
    ],
    "min_samples_leaf": [
        3,
        5,
        8,
        12,
        20,
    ],
    "alpha": [
        0.80,
        0.85,
        0.90,
        0.95,
    ],
}

search_gbr = RandomizedSearchCV(
    estimator=gbr,
    param_distributions=params_gbr,
    n_iter=18,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)

search_gbr.fit(
    X_train,
    y_train,
)

best_gbr = (
    search_gbr
    .best_estimator_
)

print(
    "\nMejores parametros:"
)

print(
    search_gbr
    .best_params_
)

print(
    f"MAE CV: "
    f"{-search_gbr.best_score_:,.2f}"
)


# ============================================================
# 19. TEST FINAL
# ============================================================

models = {
    "Extra Trees Squared":
        best_squared,

    "Extra Trees Absolute":
        best_absolute,

    "Gradient Boosting Huber":
        best_gbr,
}

predictions = pd.DataFrame(
    {
        "Date":
            test.index,

        "ActualSales":
            y_test.values,
    }
)

for name, model in models.items():

    raw_pred = (
        model.predict(
            X_test
        )
    )

    raw_pred = np.maximum(
        raw_pred,
        0,
    )

    calendar_pred = (
        apply_calendar(
            raw_pred,
            test.index,
        )
    )

    results.append(
        evaluate(
            name,
            y_test,
            raw_pred,
        )
    )

    results.append(
        evaluate(
            name + " + Calendar",
            y_test,
            calendar_pred,
        )
    )

    safe_name = (
        name.lower()
        .replace(" ", "_")
    )

    predictions[
        safe_name
    ] = raw_pred

    predictions[
        safe_name +
        "_calendar"
    ] = calendar_pred


# ============================================================
# 20. RESULTADOS
# ============================================================

results_df = (
    pd.DataFrame(
        results
    )
    .sort_values(
        "MAE"
    )
    .reset_index(
        drop=True
    )
)

print("\n" + "=" * 82)
print("RESULTADOS V3 - TEST FINAL")
print("=" * 82)

print(
    results_df
    .round(
        {
            "MAE": 2,
            "RMSE": 2,
            "R2": 4,
            "sMAPE": 2,
            "MedianAE": 2,
        }
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 21. IMPORTANCIA DE FEATURES
# ============================================================

best_calendar_name = (
    results_df[
        results_df[
            "Model"
        ].str.contains(
            "Calendar"
        )
    ]
    .iloc[0][
        "Model"
    ]
)

base_name = (
    best_calendar_name
    .replace(
        " + Calendar",
        "",
    )
)

best_model = (
    models[
        base_name
    ]
)

importance = pd.DataFrame(
    {
        "Feature":
            features,

        "Importance":
            best_model
            .feature_importances_,
    }
).sort_values(
    "Importance",
    ascending=False,
)

print("\n" + "=" * 82)
print("TOP 20 FEATURES DEL MEJOR MODELO V3")
print("=" * 82)

print(
    importance
    .head(20)
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 22. COMPARACION V1/V2/V3
# ============================================================

comparison_rows = []


v1_path = Path(
    "outputs/baseline_v1/prediction_v1.csv"
)

if v1_path.exists():

    v1 = pd.read_csv(
        v1_path
    )

    rf_v1 = (
        v1[
            v1["Model"] ==
            "Random Forest"
        ]
        .iloc[0]
    )

    comparison_rows.append(
        {
            "Version":
                "V1 Random Forest",

            "MAE":
                float(
                    rf_v1["MAE"]
                ),

            "RMSE":
                float(
                    rf_v1["RMSE"]
                ),

            "R2":
                float(
                    rf_v1["R2"]
                ),

            "sMAPE":
                float(
                    rf_v1["sMAPE"]
                ),
        }
    )


v2_path = Path(
    "outputs/prediction_v2/"
    "extra_trees_calendar_comparison.csv"
)

if v2_path.exists():

    v2 = pd.read_csv(
        v2_path
    )

    row = (
        v2[
            v2["Model"] ==
            "Extra Trees + Calendar"
        ]
        .iloc[0]
    )

    comparison_rows.append(
        {
            "Version":
                "V2 Extra Trees + Calendar",

            "MAE":
                row["MAE"],

            "RMSE":
                row["RMSE"],

            "R2":
                row["R2"],

            "sMAPE":
                row["sMAPE"],
        }
    )


best_v3 = (
    results_df[
        results_df[
            "Model"
        ].str.contains(
            "Calendar"
        )
    ]
    .iloc[0]
)

comparison_rows.append(
    {
        "Version":
            "V3 " +
            best_v3["Model"],

        "MAE":
            best_v3["MAE"],

        "RMSE":
            best_v3["RMSE"],

        "R2":
            best_v3["R2"],

        "sMAPE":
            best_v3["sMAPE"],
    }
)

comparison_df = pd.DataFrame(
    comparison_rows
)

print("\n" + "=" * 82)
print("COMPARACION V1 VS V2 VS V3")
print("=" * 82)

print(
    comparison_df
    .round(
        {
            "MAE": 2,
            "RMSE": 2,
            "R2": 4,
            "sMAPE": 2,
        }
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 23. MEJORA
# ============================================================

if len(
    comparison_df
) >= 2:

    v1_mae = (
        comparison_df
        .iloc[0][
            "MAE"
        ]
    )

    v3_mae = (
        comparison_df
        .iloc[-1][
            "MAE"
        ]
    )

    improvement = (
        (
            v1_mae -
            v3_mae
        )
        /
        v1_mae
        *
        100
    )

    print(
        f"\nMejora MAE V3 vs V1: "
        f"{improvement:.2f}%"
    )


# ============================================================
# 24. GUARDAR
# ============================================================

results_df.to_csv(
    OUTPUT_DIR /
    "model_comparison_v3.csv",
    index=False,
)

comparison_df.to_csv(
    OUTPUT_DIR /
    "version_comparison.csv",
    index=False,
)

predictions.to_csv(
    OUTPUT_DIR /
    "test_predictions_v3.csv",
    index=False,
)

importance.to_csv(
    OUTPUT_DIR /
    "best_model_feature_importance.csv",
    index=False,
)

best_params = {
    "Extra Trees Squared":
        search_squared.best_params_,

    "Extra Trees Absolute":
        search_absolute.best_params_,

    "Gradient Boosting Huber":
        search_gbr.best_params_,
}

with open(
    OUTPUT_DIR /
    "best_parameters_v3.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        best_params,
        f,
        indent=2,
    )


print(
    "\nV3 COMPLETADA."
)
