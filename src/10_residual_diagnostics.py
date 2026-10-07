from pathlib import Path
import pandas as pd
import numpy as np


PRED_FILE = Path(
    "outputs/prediction_v2/test_predictions_calendar.csv"
)

OUTPUT_DIR = Path(
    "outputs/prediction_v2"
)

print("=" * 78)
print("DIAGNOSTICO DE RESIDUOS - EXTRA TREES")
print("=" * 78)


# ============================================================
# 1. CARGAR PREDICCIONES
# ============================================================

df = pd.read_csv(
    PRED_FILE,
    parse_dates=["Date"],
)

df["DayOfWeek"] = (
    df["Date"].dt.day_name()
)

df["Month"] = (
    df["Date"].dt.month
)

df["Error"] = (
    df["ActualSales"] -
    df["extra_trees_calendar"]
)

df["AbsoluteError"] = (
    df["Error"].abs()
)

df["PercentageContribution"] = (
    df["AbsoluteError"] /
    df["AbsoluteError"].sum() *
    100
)


# ============================================================
# 2. TOP ERRORES
# ============================================================

top = (
    df.sort_values(
        "AbsoluteError",
        ascending=False,
    )
    .head(20)
    .copy()
)

print("\n" + "=" * 78)
print("TOP 20 ERRORES ABSOLUTOS")
print("=" * 78)

print(
    top[
        [
            "Date",
            "DayOfWeek",
            "ActualSales",
            "extra_trees_calendar",
            "Error",
            "AbsoluteError",
            "PercentageContribution",
        ]
    ]
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 3. CUANTO ERROR CONCENTRAN LOS OUTLIERS
# ============================================================

total_error = (
    df["AbsoluteError"].sum()
)

for n in [
    1,
    3,
    5,
    10,
    20,
]:

    contribution = (
        df.nlargest(
            n,
            "AbsoluteError",
        )["AbsoluteError"].sum()
        /
        total_error
        *
        100
    )

    print(
        f"\nTop {n:2d} dias -> "
        f"{contribution:.2f}% "
        f"del error absoluto total"
    )


# ============================================================
# 4. ERROR POR MES
# ============================================================

monthly = (
    df.groupby("Month")
    .agg(
        Days=("Date", "count"),
        ActualSales=("ActualSales", "sum"),
        PredictedSales=(
            "extra_trees_calendar",
            "sum",
        ),
        MAE=("AbsoluteError", "mean"),
        MedianAE=("AbsoluteError", "median"),
        MaxAE=("AbsoluteError", "max"),
    )
    .reset_index()
)

monthly["Bias"] = (
    monthly["PredictedSales"] -
    monthly["ActualSales"]
)

print("\n" + "=" * 78)
print("ERROR POR MES")
print("=" * 78)

print(
    monthly
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 5. ERROR POR DIA
# ============================================================

weekday = (
    df.groupby("DayOfWeek")
    .agg(
        Days=("Date", "count"),
        ActualSales=("ActualSales", "sum"),
        PredictedSales=(
            "extra_trees_calendar",
            "sum",
        ),
        MAE=("AbsoluteError", "mean"),
        MedianAE=("AbsoluteError", "median"),
        MaxAE=("AbsoluteError", "max"),
    )
    .reset_index()
)

weekday["Bias"] = (
    weekday["PredictedSales"] -
    weekday["ActualSales"]
)

order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

weekday["Order"] = (
    weekday["DayOfWeek"]
    .map(
        {
            day: i
            for i, day
            in enumerate(order)
        }
    )
)

weekday = (
    weekday
    .sort_values("Order")
    .drop(columns="Order")
)

print("\n" + "=" * 78)
print("BIAS Y ERROR POR DIA")
print("=" * 78)

print(
    weekday
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 6. GUARDAR
# ============================================================

top.to_csv(
    OUTPUT_DIR /
    "top_prediction_errors.csv",
    index=False,
)

monthly.to_csv(
    OUTPUT_DIR /
    "prediction_error_by_month.csv",
    index=False,
)

weekday.to_csv(
    OUTPUT_DIR /
    "prediction_error_by_weekday_detailed.csv",
    index=False,
)


print("\nDIAGNOSTICO COMPLETADO.")
