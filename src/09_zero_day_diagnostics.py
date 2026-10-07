from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# DIAGNOSTICO DE DIAS SIN VENTAS
# ============================================================

DATA_FILE = Path("outputs/online_retail_clean.csv")
PRED_FILE = Path("outputs/prediction_v2/test_predictions_v2.csv")

OUTPUT_DIR = Path("outputs/prediction_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_START = pd.Timestamp("2011-07-20")

print("=" * 78)
print("DIAGNOSTICO DE DIAS SIN VENTAS")
print("=" * 78)


# ------------------------------------------------------------
# 1. RECONSTRUIR SERIE DIARIA
# ------------------------------------------------------------

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["InvoiceDate"],
)

df["Date"] = df["InvoiceDate"].dt.normalize()

daily = (
    df.groupby("Date")["Sales"]
    .sum()
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

daily_df = pd.DataFrame(
    {
        "Sales": daily
    }
)

daily_df["DayOfWeek"] = (
    daily_df.index.day_name()
)

daily_df["ZeroSales"] = (
    daily_df["Sales"] == 0
)


# ------------------------------------------------------------
# 2. TRAIN Y TEST
# ------------------------------------------------------------

train = daily_df[
    daily_df.index < TEST_START
].copy()

test = daily_df[
    daily_df.index >= TEST_START
].copy()


order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def weekday_summary(data):

    summary = (
        data.groupby("DayOfWeek")
        .agg(
            Days=("Sales", "size"),
            ZeroDays=("ZeroSales", "sum"),
            MeanSales=("Sales", "mean"),
            MedianSales=("Sales", "median"),
        )
    )

    summary["ZeroRate"] = (
        summary["ZeroDays"] /
        summary["Days"]
    )

    return summary.reindex(order)


train_summary = weekday_summary(train)
test_summary = weekday_summary(test)

print("\n" + "=" * 78)
print("PATRON APRENDIDO SOLO CON TRAIN")
print("=" * 78)

print(
    train_summary
    .round(
        {
            "MeanSales": 2,
            "MedianSales": 2,
            "ZeroRate": 4,
        }
    )
    .to_string()
)


print("\n" + "=" * 78)
print("PATRON OBSERVADO EN TEST")
print("=" * 78)

print(
    test_summary
    .round(
        {
            "MeanSales": 2,
            "MedianSales": 2,
            "ZeroRate": 4,
        }
    )
    .to_string()
)


# ------------------------------------------------------------
# 3. DETECTAR DIAS ESTRUCTURALMENTE CERRADOS
# ------------------------------------------------------------
# Regla decidida usando SOLO train:
# si >=95% de ese dia historicamente tiene ventas cero,
# se considera un dia estructuralmente sin actividad.
# ------------------------------------------------------------

closed_weekdays = (
    train_summary[
        train_summary["ZeroRate"] >= 0.95
    ]
    .index
    .tolist()
)

print("\nDias estructuralmente sin ventas segun TRAIN:")

if closed_weekdays:
    for day in closed_weekdays:
        print(f"  - {day}")
else:
    print("  Ninguno")


# ------------------------------------------------------------
# 4. CARGAR PREDICCIONES V2
# ------------------------------------------------------------

pred = pd.read_csv(
    PRED_FILE,
    parse_dates=["Date"],
)

required = {
    "Date",
    "ActualSales",
    "extra_trees_tuned",
}

missing = required - set(pred.columns)

if missing:
    raise ValueError(
        f"Faltan columnas: {missing}"
    )

pred["DayOfWeek"] = (
    pred["Date"].dt.day_name()
)


# ------------------------------------------------------------
# 5. METRICAS
# ------------------------------------------------------------

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


def evaluate(name, actual, predicted):

    predicted = np.maximum(
        np.asarray(predicted),
        0,
    )

    return {
        "Model": name,
        "MAE": mean_absolute_error(
            actual,
            predicted,
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        ),
        "R2": r2_score(
            actual,
            predicted,
        ),
        "sMAPE": smape(
            actual,
            predicted,
        ),
    }


# ------------------------------------------------------------
# 6. ERROR DE EXTRA TREES POR DIA
# ------------------------------------------------------------

rows = []

for day in order:

    subset = pred[
        pred["DayOfWeek"] == day
    ]

    if len(subset) == 0:
        continue

    row = evaluate(
        day,
        subset["ActualSales"],
        subset["extra_trees_tuned"],
    )

    row["Days"] = len(subset)

    row["ZeroActualDays"] = int(
        (
            subset["ActualSales"] == 0
        ).sum()
    )

    rows.append(row)


weekday_errors = pd.DataFrame(rows)

print("\n" + "=" * 78)
print("EXTRA TREES - ERROR POR DIA DE LA SEMANA")
print("=" * 78)

print(
    weekday_errors
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


# ------------------------------------------------------------
# 7. POSTPROCESADO BASADO SOLO EN TRAIN
# ------------------------------------------------------------

pred["extra_trees_calendar"] = (
    pred["extra_trees_tuned"].copy()
)

closed_mask = pred[
    "DayOfWeek"
].isin(
    closed_weekdays
)

pred.loc[
    closed_mask,
    "extra_trees_calendar",
] = 0.0


# ------------------------------------------------------------
# 8. COMPARACION
# ------------------------------------------------------------

comparison = pd.DataFrame(
    [
        evaluate(
            "Extra Trees V2",
            pred["ActualSales"],
            pred["extra_trees_tuned"],
        ),
        evaluate(
            "Extra Trees + Calendar",
            pred["ActualSales"],
            pred["extra_trees_calendar"],
        ),
    ]
)

print("\n" + "=" * 78)
print("EXTRA TREES: ORIGINAL VS CALENDARIO")
print("=" * 78)

print(
    comparison
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


# ------------------------------------------------------------
# 9. GUARDAR
# ------------------------------------------------------------

train_summary.to_csv(
    OUTPUT_DIR /
    "weekday_zero_pattern_train.csv"
)

test_summary.to_csv(
    OUTPUT_DIR /
    "weekday_zero_pattern_test.csv"
)

weekday_errors.to_csv(
    OUTPUT_DIR /
    "extra_trees_error_by_weekday.csv",
    index=False,
)

comparison.to_csv(
    OUTPUT_DIR /
    "extra_trees_calendar_comparison.csv",
    index=False,
)

pred.to_csv(
    OUTPUT_DIR /
    "test_predictions_calendar.csv",
    index=False,
)


print("\nDIAGNOSTICO COMPLETADO.")
