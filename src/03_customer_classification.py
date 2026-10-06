from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
import joblib


# ============================================================
# TALLER 02 - MAESTRIA EN CIENCIA DE DATOS
# PARTE 1: CLASIFICACION DE CLIENTES
# ============================================================

INPUT_FILE = Path("outputs/online_retail_clean.csv")

OUTPUT_DIR = Path("outputs/classification")
FIG_DIR = Path("figures/classification")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("=" * 75)
print("TALLER 02 - CLASIFICACION DE CLIENTES")
print("=" * 75)


# ------------------------------------------------------------
# 1. CARGAR DATOS
# ------------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["InvoiceDate"]
)

print(f"\nTransacciones cargadas: {len(df):,}")
print(f"Clientes totales:       {df['CustomerID'].nunique():,}")


# ------------------------------------------------------------
# 2. DIVISION TEMPORAL
# ------------------------------------------------------------
# Usamos aproximadamente el primer año para construir las
# variables predictoras y el periodo posterior para definir
# si el cliente sera Premium.
#
# Esto evita usar informacion futura para predecir el futuro.
# ------------------------------------------------------------

cutoff = pd.Timestamp("2010-12-01")

history = df[
    df["InvoiceDate"] < cutoff
].copy()

future = df[
    df["InvoiceDate"] >= cutoff
].copy()

print("\nDivision temporal:")
print(
    f"  Historico: {history['InvoiceDate'].min()} "
    f"-> {history['InvoiceDate'].max()}"
)
print(
    f"  Futuro:    {future['InvoiceDate'].min()} "
    f"-> {future['InvoiceDate'].max()}"
)

print(f"\nClientes historicos: {history['CustomerID'].nunique():,}")
print(f"Clientes futuros:    {future['CustomerID'].nunique():,}")


# ------------------------------------------------------------
# 3. CREAR CARACTERISTICAS POR CLIENTE
# ------------------------------------------------------------

customer_features = (
    history.groupby("CustomerID")
    .agg(
        PastSales=("Sales", "sum"),
        Frequency=("Invoice", "nunique"),
        TotalQuantity=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        ActiveDays=("InvoiceDate", lambda x: x.dt.date.nunique()),
        AvgUnitPrice=("Price", "mean"),
        FirstPurchase=("InvoiceDate", "min"),
        LastPurchase=("InvoiceDate", "max"),
    )
    .reset_index()
)

customer_features["AvgItemsPerInvoice"] = (
    customer_features["TotalQuantity"] /
    customer_features["Frequency"]
)

customer_features["RecencyDays"] = (
    cutoff - customer_features["LastPurchase"]
).dt.days

customer_features["TenureDays"] = (
    customer_features["LastPurchase"] -
    customer_features["FirstPurchase"]
).dt.days + 1

customer_features = customer_features.drop(
    columns=[
        "FirstPurchase",
        "LastPurchase"
    ]
)


# ------------------------------------------------------------
# 4. DEFINIR PREMIUM CON INFORMACION FUTURA
# ------------------------------------------------------------

future_behavior = (
    future.groupby("CustomerID")
    .agg(
        FutureSales=("Sales", "sum"),
        FutureInvoices=("Invoice", "nunique"),
        FutureQuantity=("Quantity", "sum"),
    )
    .reset_index()
)

customers = customer_features.merge(
    future_behavior,
    on="CustomerID",
    how="inner"
)

print(
    f"\nClientes presentes en ambos periodos: "
    f"{len(customers):,}"
)


# ------------------------------------------------------------
# Premium = percentil superior 25% de ventas futuras
# ------------------------------------------------------------

premium_threshold = customers[
    "FutureSales"
].quantile(0.75)

customers["Premium"] = (
    customers["FutureSales"] >= premium_threshold
).astype(int)

customers["CustomerClass"] = np.where(
    customers["Premium"] == 1,
    "Premium",
    "Normal"
)

print("\n" + "=" * 75)
print("DEFINICION DE CLASES")
print("=" * 75)

print(
    f"\nUmbral Premium (P75 de ventas futuras): "
    f"£{premium_threshold:,.2f}"
)

class_counts = (
    customers["CustomerClass"]
    .value_counts()
)

print("\nDistribucion:")
print(class_counts.to_string())

print("\nPorcentajes:")
print(
    (
        customers["CustomerClass"]
        .value_counts(normalize=True) *
        100
    )
    .round(2)
    .astype(str)
    .add("%")
    .to_string()
)


# ------------------------------------------------------------
# 5. GUARDAR DATASET DE CLIENTES
# ------------------------------------------------------------

customers.to_csv(
    OUTPUT_DIR / "customer_classification_dataset.csv",
    index=False
)


# ------------------------------------------------------------
# 6. VARIABLES PREDICTORAS
# ------------------------------------------------------------

feature_columns = [
    "PastSales",
    "Frequency",
    "TotalQuantity",
    "UniqueProducts",
    "ActiveDays",
    "AvgUnitPrice",
    "AvgItemsPerInvoice",
    "RecencyDays",
    "TenureDays",
]

X = customers[feature_columns]
y = customers["Premium"]

print("\nVariables predictoras:")

for feature in feature_columns:
    print(f"  - {feature}")


# ------------------------------------------------------------
# 7. TRAIN / TEST
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nDivision del dataset:")
print(f"  Train: {len(X_train):,}")
print(f"  Test:  {len(X_test):,}")

print(
    f"  Premium train: "
    f"{y_train.mean() * 100:.2f}%"
)
print(
    f"  Premium test:  "
    f"{y_test.mean() * 100:.2f}%"
)


# ------------------------------------------------------------
# 8. BASELINE
# ------------------------------------------------------------

majority_class = y_train.mode()[0]

baseline_pred = np.full(
    len(y_test),
    majority_class
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_pred
)

print(
    f"\nBaseline accuracy "
    f"(predecir siempre clase mayoritaria): "
    f"{baseline_accuracy:.4f}"
)


# ------------------------------------------------------------
# 9. MODELO 1 - REGRESION LOGISTICA
# ------------------------------------------------------------

logistic_model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        ),
    ]
)

print("\nEntrenando Logistic Regression...")

logistic_model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# 10. MODELO 2 - RANDOM FOREST
# ------------------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("Entrenando Random Forest...")

rf_model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# 11. EVALUACION
# ------------------------------------------------------------

def evaluate_model(name, model):
    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_test,
            probabilities
        ),
    }

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Normal",
            "Premium"
        ],
        output_dict=True,
        zero_division=0
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
    )

    safe_name = (
        name.lower()
        .replace(" ", "_")
    )

    report_df.to_csv(
        OUTPUT_DIR /
        f"classification_report_{safe_name}.csv"
    )

    return (
        metrics,
        predictions,
        probabilities
    )


log_metrics, log_pred, log_prob = evaluate_model(
    "Logistic Regression",
    logistic_model
)

rf_metrics, rf_pred, rf_prob = evaluate_model(
    "Random Forest",
    rf_model
)

metrics_df = pd.DataFrame(
    [
        log_metrics,
        rf_metrics
    ]
)

metrics_df = metrics_df.sort_values(
    "F1",
    ascending=False
)

metrics_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

print("\n" + "=" * 75)
print("RESULTADOS")
print("=" * 75)

print(
    metrics_df
    .round(4)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 12. MATRICES DE CONFUSION
# ------------------------------------------------------------

for name, predictions in [
    (
        "logistic_regression",
        log_pred
    ),
    (
        "random_forest",
        rf_pred
    ),
]:
    plt.figure(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=[
            "Normal",
            "Premium"
        ],
        values_format="d"
    )

    plt.title(
        f"Matriz de confusion - "
        f"{name.replace('_', ' ').title()}"
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR /
        f"01_confusion_matrix_{name}.png",
        dpi=160
    )

    plt.close()


# ------------------------------------------------------------
# 13. CURVA ROC
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

RocCurveDisplay.from_predictions(
    y_test,
    log_prob,
    name="Logistic Regression"
)

RocCurveDisplay.from_predictions(
    y_test,
    rf_prob,
    name="Random Forest",
    ax=plt.gca()
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Clasificador aleatorio"
)

plt.title(
    "Curvas ROC - Clasificacion Premium"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR / "02_roc_curve.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# 14. IMPORTANCIA DE VARIABLES - RANDOM FOREST
# ------------------------------------------------------------

importance = pd.DataFrame(
    {
        "Feature": feature_columns,
        "Importance": rf_model.feature_importances_
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

print("\n" + "=" * 75)
print("IMPORTANCIA DE VARIABLES - RANDOM FOREST")
print("=" * 75)

print(
    importance
    .round(4)
    .to_string(index=False)
)

plot_importance = (
    importance
    .sort_values("Importance")
)

plt.figure(figsize=(9, 6))

plt.barh(
    plot_importance["Feature"],
    plot_importance["Importance"]
)

plt.xlabel("Importancia")
plt.ylabel("Variable")

plt.title(
    "Importancia de variables - Random Forest"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "03_random_forest_feature_importance.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# 15. COEFICIENTES REGRESION LOGISTICA
# ------------------------------------------------------------

log_coefficients = pd.DataFrame(
    {
        "Feature": feature_columns,
        "Coefficient":
            logistic_model[
                "model"
            ].coef_[0]
    }
)

log_coefficients[
    "AbsCoefficient"
] = log_coefficients[
    "Coefficient"
].abs()

log_coefficients = (
    log_coefficients
    .sort_values(
        "AbsCoefficient",
        ascending=False
    )
)

log_coefficients.to_csv(
    OUTPUT_DIR /
    "logistic_regression_coefficients.csv",
    index=False
)


# ------------------------------------------------------------
# 16. GUARDAR MODELOS
# ------------------------------------------------------------

joblib.dump(
    logistic_model,
    MODEL_DIR /
    "logistic_customer_classifier.joblib"
)

joblib.dump(
    rf_model,
    MODEL_DIR /
    "random_forest_customer_classifier.joblib"
)

best_model_name = (
    metrics_df.iloc[0]["Model"]
)

print(
    f"\nMejor modelo segun F1: "
    f"{best_model_name}"
)


# ------------------------------------------------------------
# 17. RESUMEN
# ------------------------------------------------------------

summary_lines = [
    "TALLER 02 - CLASIFICACION DE CLIENTES",
    "=" * 60,
    "",
    "Definicion:",
    (
        "Premium = cliente situado en el 25% superior "
        "de ventas del periodo futuro."
    ),
    "",
    f"Fecha de corte: {cutoff.date()}",
    (
        f"Clientes utilizados: "
        f"{len(customers):,}"
    ),
    (
        f"Umbral Premium: "
        f"GBP {premium_threshold:,.2f}"
    ),
    (
        f"Premium: "
        f"{int(y.sum()):,}"
    ),
    (
        f"Normales: "
        f"{int((1-y).sum()):,}"
    ),
    "",
    (
        f"Baseline accuracy: "
        f"{baseline_accuracy:.4f}"
    ),
    "",
    "Resultados:",
    metrics_df.round(4).to_string(
        index=False
    ),
    "",
    (
        f"Mejor modelo por F1: "
        f"{best_model_name}"
    ),
]

(
    OUTPUT_DIR /
    "CLASSIFICATION_SUMMARY.txt"
).write_text(
    "\n".join(summary_lines),
    encoding="utf-8"
)


print("\nArchivos generados:")

for file in sorted(
    OUTPUT_DIR.glob("*")
):
    print(f"  {file}")

print("\nFiguras:")

for file in sorted(
    FIG_DIR.glob("*.png")
):
    print(f"  {file}")

print("\nModelos:")

for file in sorted(
    MODEL_DIR.glob(
        "*customer_classifier.joblib"
    )
):
    print(f"  {file}")

print(
    "\nPARTE 1 COMPLETADA CORRECTAMENTE."
)
