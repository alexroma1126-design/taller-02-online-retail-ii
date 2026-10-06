from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.base import clone
import joblib


# ============================================================
# TALLER 02 - MAESTRIA EN CIENCIA DE DATOS
# PARTE 3: PREDICCION DE VENTAS
# ============================================================

INPUT_FILE = Path("outputs/online_retail_clean.csv")

OUTPUT_DIR = Path("outputs/prediction")
FIG_DIR = Path("figures/prediction")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("=" * 78)
print("TALLER 02 - PREDICCION DE VENTAS")
print("=" * 78)


# ------------------------------------------------------------
# 1. CARGAR TRANSACCIONES
# ------------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["InvoiceDate"]
)

print(f"\nTransacciones: {len(df):,}")
print(
    f"Periodo: "
    f"{df['InvoiceDate'].min()} -> "
    f"{df['InvoiceDate'].max()}"
)


# ------------------------------------------------------------
# 2. AGREGAR VENTAS POR DIA
# ------------------------------------------------------------

df["Date"] = (
    df["InvoiceDate"]
    .dt.normalize()
)

daily = (
    df.groupby("Date")
    .agg(
        Sales=("Sales", "sum"),
        Quantity=("Quantity", "sum"),
        Invoices=("Invoice", "nunique"),
        Customers=("CustomerID", "nunique")
    )
    .sort_index()
)


# ------------------------------------------------------------
# Incluir dias sin transacciones
# ------------------------------------------------------------

full_calendar = pd.date_range(
    daily.index.min(),
    daily.index.max(),
    freq="D"
)

daily = (
    daily
    .reindex(full_calendar)
    .fillna(0)
)

daily.index.name = "Date"

print(
    f"\nDias calendario: "
    f"{len(daily):,}"
)

print(
    f"Dias con ventas: "
    f"{(daily['Sales'] > 0).sum():,}"
)

print(
    f"Dias sin ventas: "
    f"{(daily['Sales'] == 0).sum():,}"
)


# ------------------------------------------------------------
# 3. VARIABLES TEMPORALES
# ------------------------------------------------------------

daily["DayOfWeek"] = daily.index.dayofweek
daily["Month"] = daily.index.month
daily["DayOfMonth"] = daily.index.day
daily["DayOfYear"] = daily.index.dayofyear

daily["IsWeekend"] = (
    daily["DayOfWeek"] >= 5
).astype(int)


# ------------------------------------------------------------
# Variables ciclicas
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 4. REZAGOS
# ------------------------------------------------------------
# Todas las variables de ventas usan shift:
# nunca utilizamos las ventas del dia objetivo
# para predecir ese mismo dia.
# ------------------------------------------------------------

for lag in [
    1,
    7,
    14,
    28
]:
    daily[
        f"SalesLag{lag}"
    ] = daily[
        "Sales"
    ].shift(lag)


# ------------------------------------------------------------
# 5. MEDIAS MOVILES HISTORICAS
# ------------------------------------------------------------

daily["SalesRolling7"] = (
    daily["Sales"]
    .shift(1)
    .rolling(7)
    .mean()
)

daily["SalesRolling14"] = (
    daily["Sales"]
    .shift(1)
    .rolling(14)
    .mean()
)

daily["SalesRolling28"] = (
    daily["Sales"]
    .shift(1)
    .rolling(28)
    .mean()
)

daily["SalesStd7"] = (
    daily["Sales"]
    .shift(1)
    .rolling(7)
    .std()
)

daily["SalesStd28"] = (
    daily["Sales"]
    .shift(1)
    .rolling(28)
    .std()
)


# ------------------------------------------------------------
# 6. DATASET DE MODELADO
# ------------------------------------------------------------

feature_columns = [
    "DayOfWeek",
    "Month",
    "DayOfMonth",
    "DayOfYear",
    "IsWeekend",
    "DowSin",
    "DowCos",
    "YearSin",
    "YearCos",
    "SalesLag1",
    "SalesLag7",
    "SalesLag14",
    "SalesLag28",
    "SalesRolling7",
    "SalesRolling14",
    "SalesRolling28",
    "SalesStd7",
    "SalesStd28",
]

model_data = (
    daily[
        ["Sales"] +
        feature_columns
    ]
    .dropna()
    .copy()
)

print(
    f"\nObservaciones utilizables: "
    f"{len(model_data):,}"
)

print("\nVariables predictoras:")

for variable in feature_columns:
    print(f"  - {variable}")


# ------------------------------------------------------------
# 7. SPLIT CRONOLOGICO
# ------------------------------------------------------------
# 80% pasado -> entrenamiento
# 20% final -> prueba futura
# ------------------------------------------------------------

split_index = int(
    len(model_data) * 0.80
)

train = model_data.iloc[
    :split_index
].copy()

test = model_data.iloc[
    split_index:
].copy()

X_train = train[
    feature_columns
]

y_train = train[
    "Sales"
]

X_test = test[
    feature_columns
]

y_test = test[
    "Sales"
]

print("\nDivision temporal:")

print(
    f"Train: "
    f"{train.index.min().date()} "
    f"-> {train.index.max().date()} "
    f"({len(train)} dias)"
)

print(
    f"Test:  "
    f"{test.index.min().date()} "
    f"-> {test.index.max().date()} "
    f"({len(test)} dias)"
)


# ------------------------------------------------------------
# 8. MODELOS
# ------------------------------------------------------------

models = {
    "Linear Regression":
        Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    LinearRegression()
                )
            ]
        ),

    "Decision Tree":
        DecisionTreeRegressor(
            max_depth=8,
            min_samples_leaf=5,
            random_state=RANDOM_STATE
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
}


# ------------------------------------------------------------
# 9. VALIDACION TEMPORAL
# ------------------------------------------------------------

print("\n" + "=" * 78)
print("VALIDACION TEMPORAL")
print("=" * 78)

tscv = TimeSeriesSplit(
    n_splits=5
)

cv_rows = []

for name, model in models.items():

    fold_maes = []

    for fold, (
        train_idx,
        valid_idx
    ) in enumerate(
        tscv.split(X_train),
        start=1
    ):

        fold_model = clone(model)

        fold_model.fit(
            X_train.iloc[
                train_idx
            ],
            y_train.iloc[
                train_idx
            ]
        )

        fold_pred = (
            fold_model.predict(
                X_train.iloc[
                    valid_idx
                ]
            )
        )

        mae = mean_absolute_error(
            y_train.iloc[
                valid_idx
            ],
            fold_pred
        )

        fold_maes.append(
            mae
        )

        cv_rows.append(
            {
                "Model": name,
                "Fold": fold,
                "MAE": mae
            }
        )

    print(
        f"{name}: "
        f"MAE CV = "
        f"£{np.mean(fold_maes):,.2f} "
        f"+/- "
        f"£{np.std(fold_maes):,.2f}"
    )


cv_df = pd.DataFrame(
    cv_rows
)

cv_df.to_csv(
    OUTPUT_DIR /
    "time_series_cv_results.csv",
    index=False
)


# ------------------------------------------------------------
# 10. METRICAS
# ------------------------------------------------------------

def smape(
    actual,
    predicted
):

    actual = np.asarray(
        actual
    )

    predicted = np.asarray(
        predicted
    )

    denominator = (
        np.abs(actual) +
        np.abs(predicted)
    )

    mask = denominator != 0

    return (
        100 *
        np.mean(
            2 *
            np.abs(
                predicted[mask] -
                actual[mask]
            ) /
            denominator[mask]
        )
    )


def evaluate(
    name,
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "sMAPE": smape(
            actual,
            predicted
        )
    }


# ------------------------------------------------------------
# 11. BASELINE
# ------------------------------------------------------------
# Prediccion ingenua:
# ventas de hoy = ventas de ayer
# ------------------------------------------------------------

baseline_pred = (
    X_test[
        "SalesLag1"
    ]
    .to_numpy()
)

results = [
    evaluate(
        "Baseline Lag-1",
        y_test,
        baseline_pred
    )
]

predictions_df = pd.DataFrame(
    {
        "Date": test.index,
        "ActualSales": y_test.values,
        "Baseline": baseline_pred
    }
)


# ------------------------------------------------------------
# 12. ENTRENAR Y EVALUAR MODELOS
# ------------------------------------------------------------

trained_models = {}

print("\n" + "=" * 78)
print("RESULTADOS EN TEST FUTURO")
print("=" * 78)

for name, model in models.items():

    print(
        f"\nEntrenando "
        f"{name}..."
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    predictions = np.maximum(
        predictions,
        0
    )

    trained_models[
        name
    ] = model

    results.append(
        evaluate(
            name,
            y_test,
            predictions
        )
    )

    safe_name = (
        name.lower()
        .replace(
            " ",
            "_"
        )
    )

    predictions_df[
        safe_name
    ] = predictions


# ------------------------------------------------------------
# 13. TABLA COMPARATIVA
# ------------------------------------------------------------

results_df = pd.DataFrame(
    results
)

results_df = (
    results_df
    .sort_values(
        "MAE"
    )
    .reset_index(
        drop=True
    )
)

results_df.to_csv(
    OUTPUT_DIR /
    "model_comparison.csv",
    index=False
)

print(
    "\n" +
    results_df
    .round(
        {
            "MAE": 2,
            "RMSE": 2,
            "R2": 4,
            "sMAPE": 2
        }
    )
    .to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 14. MEJOR MODELO ML
# ------------------------------------------------------------

ml_results = results_df[
    results_df["Model"] !=
    "Baseline Lag-1"
]

best_model_name = (
    ml_results.iloc[0][
        "Model"
    ]
)

best_model = trained_models[
    best_model_name
]

print(
    f"\nMejor modelo ML "
    f"segun MAE: "
    f"{best_model_name}"
)


# ------------------------------------------------------------
# 15. GUARDAR MODELOS
# ------------------------------------------------------------

for name, model in trained_models.items():

    safe_name = (
        name.lower()
        .replace(
            " ",
            "_"
        )
    )

    joblib.dump(
        model,
        MODEL_DIR /
        f"{safe_name}_sales_forecaster.joblib"
    )


# ------------------------------------------------------------
# 16. GUARDAR PREDICCIONES
# ------------------------------------------------------------

predictions_df.to_csv(
    OUTPUT_DIR /
    "test_predictions.csv",
    index=False
)

model_data.reset_index().to_csv(
    OUTPUT_DIR /
    "daily_sales_model_dataset.csv",
    index=False
)


# ------------------------------------------------------------
# 17. SERIE COMPLETA
# ------------------------------------------------------------

plt.figure(
    figsize=(13, 6)
)

plt.plot(
    daily.index,
    daily["Sales"],
    linewidth=1
)

plt.xlabel("Fecha")
plt.ylabel("Ventas diarias (£)")
plt.title(
    "Serie temporal de ventas diarias"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "01_daily_sales_series.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# 18. REAL VS PREDICCION
# ------------------------------------------------------------

best_prediction_column = (
    best_model_name.lower()
    .replace(
        " ",
        "_"
    )
)

plt.figure(
    figsize=(13, 6)
)

plt.plot(
    predictions_df[
        "Date"
    ],
    predictions_df[
        "ActualSales"
    ],
    label="Real"
)

plt.plot(
    predictions_df[
        "Date"
    ],
    predictions_df[
        best_prediction_column
    ],
    label=best_model_name
)

plt.xlabel("Fecha")
plt.ylabel("Ventas diarias (£)")

plt.title(
    "Ventas reales vs predichas "
    f"- {best_model_name}"
)

plt.legend()
plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "02_actual_vs_predicted.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# 19. IMPORTANCIA RANDOM FOREST
# ------------------------------------------------------------

rf_model = trained_models[
    "Random Forest"
]

importance = pd.DataFrame(
    {
        "Feature":
            feature_columns,
        "Importance":
            rf_model.feature_importances_
    }
).sort_values(
    "Importance",
    ascending=False
)

importance.to_csv(
    OUTPUT_DIR /
    "random_forest_feature_importance.csv",
    index=False
)

print("\n" + "=" * 78)
print("IMPORTANCIA DE VARIABLES - RANDOM FOREST")
print("=" * 78)

print(
    importance
    .round(4)
    .to_string(
        index=False
    )
)

plot_importance = (
    importance
    .sort_values(
        "Importance"
    )
)

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    plot_importance[
        "Feature"
    ],
    plot_importance[
        "Importance"
    ]
)

plt.xlabel("Importancia")
plt.ylabel("Variable")

plt.title(
    "Importancia de variables "
    "- Random Forest"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "03_random_forest_feature_importance.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# 20. RESIDUOS DEL MEJOR MODELO
# ------------------------------------------------------------

actual = predictions_df[
    "ActualSales"
].to_numpy()

predicted = predictions_df[
    best_prediction_column
].to_numpy()

residuals = (
    actual -
    predicted
)

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    predicted,
    residuals,
    alpha=0.6
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Ventas predichas (£)"
)

plt.ylabel(
    "Residuo: real - predicho (£)"
)

plt.title(
    f"Analisis de residuos "
    f"- {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "04_prediction_residuals.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# 21. RESUMEN
# ------------------------------------------------------------

summary_lines = [
    "TALLER 02 - PREDICCION DE VENTAS",
    "=" * 65,
    "",
    (
        "Unidad temporal: "
        "ventas diarias"
    ),
    (
        f"Observaciones modeladas: "
        f"{len(model_data):,}"
    ),
    "",
    (
        f"Train: "
        f"{train.index.min().date()} "
        f"-> "
        f"{train.index.max().date()}"
    ),
    (
        f"Test: "
        f"{test.index.min().date()} "
        f"-> "
        f"{test.index.max().date()}"
    ),
    "",
    "Variables predictoras:",
    *[
        f"- {x}"
        for x in feature_columns
    ],
    "",
    "RESULTADOS",
    results_df
    .round(4)
    .to_string(
        index=False
    ),
    "",
    (
        f"Mejor modelo ML "
        f"segun MAE: "
        f"{best_model_name}"
    )
]

(
    OUTPUT_DIR /
    "PREDICTION_SUMMARY.txt"
).write_text(
    "\n".join(
        summary_lines
    ),
    encoding="utf-8"
)


print("\n" + "=" * 78)
print("ARCHIVOS GENERADOS")
print("=" * 78)

print("\nTablas:")

for file in sorted(
    OUTPUT_DIR.glob("*")
):
    print(
        f"  {file}"
    )

print("\nFiguras:")

for file in sorted(
    FIG_DIR.glob("*.png")
):
    print(
        f"  {file}"
    )

print(
    "\nPARTE 3 COMPLETADA "
    "CORRECTAMENTE."
)
