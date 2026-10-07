from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
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
# TALLER 02 - PREDICCION DE VENTAS V2
# ============================================================

INPUT_FILE = Path("outputs/online_retail_clean.csv")

OUTPUT_DIR = Path("outputs/prediction_v2")
FIG_DIR = Path("figures/prediction_v2")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("=" * 78)
print("PREDICCION DE VENTAS V2")
print("=" * 78)


# ============================================================
# 1. DATOS
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["InvoiceDate"],
)

df["Date"] = df["InvoiceDate"].dt.normalize()

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

daily = daily.reindex(calendar).fillna(0)
daily.index.name = "Date"

print(f"\nDias calendario: {len(daily):,}")
print(f"Dias con ventas: {(daily['Sales'] > 0).sum():,}")
print(f"Dias sin ventas: {(daily['Sales'] == 0).sum():,}")


# ============================================================
# 2. FEATURES DE CALENDARIO
# ============================================================

daily["DayOfWeek"] = daily.index.dayofweek
daily["Month"] = daily.index.month
daily["DayOfMonth"] = daily.index.day
daily["DayOfYear"] = daily.index.dayofyear
daily["WeekOfYear"] = daily.index.isocalendar().week.astype(int)
daily["Quarter"] = daily.index.quarter

daily["IsWeekend"] = (
    daily["DayOfWeek"] >= 5
).astype(int)

daily["IsMonthStart"] = (
    daily.index.is_month_start
).astype(int)

daily["IsMonthEnd"] = (
    daily.index.is_month_end
).astype(int)

daily["PeakSeason"] = (
    daily["Month"].isin([9, 10, 11])
).astype(int)

daily["IsDecember"] = (
    daily["Month"] == 12
).astype(int)


# ============================================================
# 3. VARIABLES CICLICAS
# ============================================================

daily["DowSin"] = np.sin(
    2 * np.pi * daily["DayOfWeek"] / 7
)

daily["DowCos"] = np.cos(
    2 * np.pi * daily["DayOfWeek"] / 7
)

daily["YearSin"] = np.sin(
    2 * np.pi * daily["DayOfYear"] / 365.25
)

daily["YearCos"] = np.cos(
    2 * np.pi * daily["DayOfYear"] / 365.25
)


# ============================================================
# 4. LAGS
# ============================================================

lags = [
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

for lag in lags:
    daily[f"SalesLag{lag}"] = (
        daily["Sales"].shift(lag)
    )


# ============================================================
# 5. VENTANAS MOVILES
# ============================================================

past_sales = daily["Sales"].shift(1)

for window in [7, 14, 28]:

    daily[f"RollingMean{window}"] = (
        past_sales
        .rolling(window)
        .mean()
    )

    daily[f"RollingMedian{window}"] = (
        past_sales
        .rolling(window)
        .median()
    )

    daily[f"RollingStd{window}"] = (
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
# 6. DATASET
# ============================================================

calendar_features = [
    "DayOfWeek",
    "Month",
    "DayOfMonth",
    "DayOfYear",
    "WeekOfYear",
    "Quarter",
    "IsWeekend",
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
    for x in lags
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

features = (
    calendar_features +
    lag_features +
    rolling_features
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
    f"Features: {len(features)}"
)


# ============================================================
# 7. SPLIT FINAL CRONOLOGICO
# ============================================================

split_index = int(
    len(model_data) * 0.80
)

train = model_data.iloc[
    :split_index
].copy()

test = model_data.iloc[
    split_index:
].copy()

X_train = train[features]
y_train = train["Sales"]

X_test = test[features]
y_test = test["Sales"]

print("\nSplit final:")

print(
    f"Train: {train.index.min().date()} "
    f"-> {train.index.max().date()} "
    f"({len(train)} dias)"
)

print(
    f"Test:  {test.index.min().date()} "
    f"-> {test.index.max().date()} "
    f"({len(test)} dias)"
)


# ============================================================
# 8. METRICAS
# ============================================================

def smape(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = (
        np.abs(y_true) +
        np.abs(y_pred)
    )

    valid = denominator != 0

    return (
        100 *
        np.mean(
            2 *
            np.abs(
                y_pred[valid] -
                y_true[valid]
            ) /
            denominator[valid]
        )
    )


def evaluate(
    name,
    y_true,
    y_pred,
):

    y_pred = np.maximum(
        np.asarray(y_pred),
        0,
    )

    return {
        "Model": name,
        "MAE": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        ),
        "R2": r2_score(
            y_true,
            y_pred,
        ),
        "sMAPE": smape(
            y_true,
            y_pred,
        ),
    }


# ============================================================
# 9. BASELINES FUERTES
# ============================================================

results = []

baseline_predictions = {
    "Baseline Lag-1":
        X_test["SalesLag1"].values,

    "Baseline Lag-7":
        X_test["SalesLag7"].values,

    "Baseline Rolling-7":
        X_test["RollingMean7"].values,
}

print("\n" + "=" * 78)
print("BASELINES")
print("=" * 78)

for name, prediction in baseline_predictions.items():

    row = evaluate(
        name,
        y_test,
        prediction,
    )

    results.append(row)

    print(
        f"{name}: "
        f"MAE={row['MAE']:,.2f} | "
        f"R2={row['R2']:.4f} | "
        f"sMAPE={row['sMAPE']:.2f}"
    )


# ============================================================
# 10. TIME SERIES CV
# ============================================================

tscv = TimeSeriesSplit(
    n_splits=5
)


# ============================================================
# 11. RANDOM FOREST OPTIMIZADO
# ============================================================

print("\n" + "=" * 78)
print("TUNING RANDOM FOREST")
print("=" * 78)

rf = RandomForestRegressor(
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

rf_params = {
    "n_estimators": [
        250,
        400,
        600,
        800,
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
        8,
        12,
    ],
    "min_samples_leaf": [
        1,
        2,
        4,
        8,
    ],
    "max_features": [
        1.0,
        "sqrt",
        0.5,
        0.7,
    ],
}

rf_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_params,
    n_iter=16,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)

rf_search.fit(
    X_train,
    y_train,
)

best_rf = rf_search.best_estimator_

print(
    "\nMejores parametros RF:"
)

print(
    rf_search.best_params_
)

print(
    "MAE CV RF: "
    f"{-rf_search.best_score_:,.2f}"
)


# ============================================================
# 12. EXTRA TREES
# ============================================================

print("\n" + "=" * 78)
print("TUNING EXTRA TREES")
print("=" * 78)

et = ExtraTreesRegressor(
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

et_params = {
    "n_estimators": [
        250,
        400,
        600,
        800,
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
        8,
        12,
    ],
    "min_samples_leaf": [
        1,
        2,
        4,
        8,
    ],
    "max_features": [
        1.0,
        "sqrt",
        0.5,
        0.7,
    ],
}

et_search = RandomizedSearchCV(
    estimator=et,
    param_distributions=et_params,
    n_iter=16,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)

et_search.fit(
    X_train,
    y_train,
)

best_et = et_search.best_estimator_

print(
    "\nMejores parametros Extra Trees:"
)

print(
    et_search.best_params_
)

print(
    "MAE CV Extra Trees: "
    f"{-et_search.best_score_:,.2f}"
)


# ============================================================
# 13. HIST GRADIENT BOOSTING
# ============================================================

print("\n" + "=" * 78)
print("TUNING HIST GRADIENT BOOSTING")
print("=" * 78)

hgb = HistGradientBoostingRegressor(
    random_state=RANDOM_STATE,
)

hgb_params = {
    "learning_rate": [
        0.03,
        0.05,
        0.08,
        0.10,
    ],
    "max_iter": [
        100,
        200,
        300,
        400,
    ],
    "max_leaf_nodes": [
        7,
        15,
        31,
        63,
    ],
    "max_depth": [
        None,
        3,
        5,
        7,
    ],
    "min_samples_leaf": [
        5,
        10,
        20,
        30,
    ],
    "l2_regularization": [
        0.0,
        0.1,
        1.0,
        5.0,
    ],
}

hgb_search = RandomizedSearchCV(
    estimator=hgb,
    param_distributions=hgb_params,
    n_iter=20,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)

hgb_search.fit(
    X_train,
    y_train,
)

best_hgb = hgb_search.best_estimator_

print(
    "\nMejores parametros HGB:"
)

print(
    hgb_search.best_params_
)

print(
    "MAE CV HGB: "
    f"{-hgb_search.best_score_:,.2f}"
)


# ============================================================
# 14. TEST FINAL
# ============================================================

models = {
    "Random Forest Tuned":
        best_rf,

    "Extra Trees Tuned":
        best_et,

    "HistGradientBoosting":
        best_hgb,
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

    pred = model.predict(
        X_test
    )

    pred = np.maximum(
        pred,
        0,
    )

    results.append(
        evaluate(
            name,
            y_test,
            pred,
        )
    )

    safe_name = (
        name.lower()
        .replace(" ", "_")
    )

    predictions[
        safe_name
    ] = pred


# ============================================================
# 15. RESULTADOS
# ============================================================

results_df = (
    pd.DataFrame(results)
    .sort_values("MAE")
    .reset_index(drop=True)
)

results_df.to_csv(
    OUTPUT_DIR /
    "model_comparison_v2.csv",
    index=False,
)

predictions.to_csv(
    OUTPUT_DIR /
    "test_predictions_v2.csv",
    index=False,
)

print("\n" + "=" * 78)
print("RESULTADOS TEST FINAL")
print("=" * 78)

print(
    results_df
    .round(
        {
            "MAE": 2,
            "RMSE": 2,
            "R2": 4,
            "sMAPE": 2,
        }
    )
    .to_string(index=False)
)


# ============================================================
# 16. GUARDAR PARAMETROS
# ============================================================

params = {
    "Random Forest Tuned":
        rf_search.best_params_,

    "Extra Trees Tuned":
        et_search.best_params_,

    "HistGradientBoosting":
        hgb_search.best_params_,
}

with open(
    OUTPUT_DIR /
    "best_parameters.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        params,
        f,
        indent=2,
    )


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

for model_name, model in [
    ("random_forest", best_rf),
    ("extra_trees", best_et),
]:

    importance = pd.DataFrame(
        {
            "Feature":
                features,

            "Importance":
                model.feature_importances_,
        }
    ).sort_values(
        "Importance",
        ascending=False,
    )

    importance.to_csv(
        OUTPUT_DIR /
        f"{model_name}_feature_importance.csv",
        index=False,
    )

    print(
        f"\nTOP 10 FEATURES - "
        f"{model_name.upper()}"
    )

    print(
        importance
        .head(10)
        .round(4)
        .to_string(index=False)
    )


# ============================================================
# 18. GRAFICO MEJOR MODELO
# ============================================================

best_ml = (
    results_df[
        ~results_df["Model"]
        .str.startswith("Baseline")
    ]
    .iloc[0]
)

best_name = best_ml[
    "Model"
]

best_col = (
    best_name.lower()
    .replace(" ", "_")
)

plt.figure(
    figsize=(13, 6)
)

plt.plot(
    predictions["Date"],
    predictions["ActualSales"],
    label="Real",
)

plt.plot(
    predictions["Date"],
    predictions[best_col],
    label=best_name,
)

plt.xlabel("Fecha")
plt.ylabel("Ventas diarias")
plt.title(
    f"Ventas reales vs predichas - {best_name}"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "01_best_model_actual_vs_predicted.png",
    dpi=160,
)

plt.close()


# ============================================================
# 19. COMPARACION MAE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plot_df = (
    results_df
    .sort_values(
        "MAE",
        ascending=True,
    )
)

plt.barh(
    plot_df["Model"],
    plot_df["MAE"],
)

plt.xlabel("MAE")
plt.ylabel("Modelo")
plt.title(
    "Comparacion de MAE - V2"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "02_model_mae_comparison.png",
    dpi=160,
)

plt.close()


# ============================================================
# 20. COMPARAR CONTRA V1
# ============================================================

v1_path = Path(
    "outputs/baseline_v1/prediction_v1.csv"
)

if v1_path.exists():

    v1 = pd.read_csv(
        v1_path
    )

    rf_v1 = v1[
        v1["Model"] ==
        "Random Forest"
    ].iloc[0]

    best_v2 = (
        results_df[
            ~results_df["Model"]
            .str.startswith("Baseline")
        ]
        .iloc[0]
    )

    improvement = (
        rf_v1["MAE"] -
        best_v2["MAE"]
    ) / rf_v1["MAE"] * 100

    print("\n" + "=" * 78)
    print("V1 VS V2")
    print("=" * 78)

    print(
        f"Random Forest V1 MAE: "
        f"{rf_v1['MAE']:,.2f}"
    )

    print(
        f"Mejor V2: "
        f"{best_v2['Model']}"
    )

    print(
        f"Mejor V2 MAE: "
        f"{best_v2['MAE']:,.2f}"
    )

    print(
        f"Mejora V2 vs V1: "
        f"{improvement:.2f}%"
    )


print("\nV2 COMPLETADA.")
