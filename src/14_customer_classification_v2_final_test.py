from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# CLASIFICACION V2 - TEST FINAL
# ============================================================

DATA_FILE = Path(
    "outputs/classification_v2/customer_classification_dataset.csv"
)

SPLIT_FILE = Path(
    "outputs/classification_v2/classification_split.csv"
)

META_FILE = Path(
    "outputs/classification_v2/selection_metadata.json"
)

MODEL_FILE = Path(
    "models/classification_v2_candidate.joblib"
)

V1_FILE = Path(
    "outputs/baseline_v1/classification_v1.csv"
)

OUTPUT_DIR = Path(
    "outputs/classification_v2"
)

print("=" * 82)
print("CLASIFICACION V2 - APERTURA UNICA DEL TEST FINAL")
print("=" * 82)


# ============================================================
# 1. CARGAR
# ============================================================

customers = pd.read_csv(
    DATA_FILE
)

split = pd.read_csv(
    SPLIT_FILE
)

with open(
    META_FILE,
    "r",
    encoding="utf-8",
) as f:

    metadata = json.load(f)


model = joblib.load(
    MODEL_FILE
)


# ============================================================
# 2. VARIABLES
# ============================================================

features = [
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


# ============================================================
# 3. RECUPERAR EXACTAMENTE EL TEST RESERVADO
# ============================================================

data = customers.merge(
    split,
    on="CustomerID",
    how="inner",
)

test = data[
    data["Split"] == "test"
].copy()

X_test = test[
    features
]

y_test = test[
    "Premium"
]

threshold = float(
    metadata[
        "champion_threshold"
    ]
)

champion_name = metadata[
    "champion_model"
]


print(
    f"\nModelo seleccionado previamente: "
    f"{champion_name}"
)

print(
    f"Threshold fijado previamente: "
    f"{threshold:.2f}"
)

print(
    f"Clientes del test: "
    f"{len(test):,}"
)

print(
    f"Premium reales: "
    f"{int(y_test.sum()):,}"
)

print(
    f"Normales reales: "
    f"{int((1-y_test).sum()):,}"
)


# ============================================================
# 4. ABRIR TEST
# ============================================================

probabilities = (
    model.predict_proba(
        X_test
    )[:, 1]
)

predictions = (
    probabilities >= threshold
).astype(int)


# ============================================================
# 5. METRICAS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions,
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    probabilities,
)

pr_auc = average_precision_score(
    y_test,
    probabilities,
)

tn, fp, fn, tp = (
    confusion_matrix(
        y_test,
        predictions,
    )
    .ravel()
)


# ============================================================
# 6. RESULTADO V2
# ============================================================

result = pd.DataFrame(
    [
        {
            "Model":
                champion_name,

            "Threshold":
                threshold,

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1":
                f1,

            "ROC_AUC":
                roc_auc,

            "PR_AUC":
                pr_auc,

            "TN":
                tn,

            "FP":
                fp,

            "FN":
                fn,

            "TP":
                tp,
        }
    ]
)


print("\n" + "=" * 82)
print("RESULTADO FINAL V2")
print("=" * 82)

print(
    result
    .round(
        {
            "Threshold": 2,
            "Accuracy": 4,
            "Precision": 4,
            "Recall": 4,
            "F1": 4,
            "ROC_AUC": 4,
            "PR_AUC": 4,
        }
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 7. COMPARAR CONTRA V1 YA CONOCIDA
# ============================================================

if V1_FILE.exists():

    v1 = pd.read_csv(
        V1_FILE
    )

    rf_v1 = (
        v1[
            v1["Model"] ==
            "Random Forest"
        ]
        .iloc[0]
    )

    comparison = pd.DataFrame(
        [
            {
                "Version":
                    "V1 Random Forest",

                "Accuracy":
                    rf_v1["Accuracy"],

                "Precision":
                    rf_v1["Precision"],

                "Recall":
                    rf_v1["Recall"],

                "F1":
                    rf_v1["F1"],

                "ROC_AUC":
                    rf_v1["ROC_AUC"],
            },

            {
                "Version":
                    (
                        "V2 " +
                        champion_name
                    ),

                "Accuracy":
                    accuracy,

                "Precision":
                    precision,

                "Recall":
                    recall,

                "F1":
                    f1,

                "ROC_AUC":
                    roc_auc,
            },
        ]
    )

    print("\n" + "=" * 82)
    print("V1 VS V2 - TEST FINAL")
    print("=" * 82)

    print(
        comparison
        .round(4)
        .to_string(
            index=False
        )
    )

    f1_improvement = (
        (
            f1 -
            float(
                rf_v1["F1"]
            )
        )
        /
        float(
            rf_v1["F1"]
        )
        *
        100
    )

    auc_change = (
        roc_auc -
        float(
            rf_v1["ROC_AUC"]
        )
    )

    print(
        f"\nCambio relativo de F1 "
        f"V2 vs V1: "
        f"{f1_improvement:.2f}%"
    )

    print(
        f"Cambio absoluto ROC-AUC "
        f"V2 vs V1: "
        f"{auc_change:+.4f}"
    )

    comparison.to_csv(
        OUTPUT_DIR /
        "v1_vs_v2_final_comparison.csv",
        index=False,
    )


# ============================================================
# 8. GUARDAR PREDICCIONES
# ============================================================

pred_output = pd.DataFrame(
    {
        "CustomerID":
            test["CustomerID"].values,

        "ActualPremium":
            y_test.values,

        "PremiumProbability":
            probabilities,

        "PredictedPremium":
            predictions,
    }
)

pred_output.to_csv(
    OUTPUT_DIR /
    "final_test_predictions.csv",
    index=False,
)

result.to_csv(
    OUTPUT_DIR /
    "final_test_metrics.csv",
    index=False,
)


# ============================================================
# 9. MARCAR TEST COMO ABIERTO
# ============================================================

metadata[
    "test_evaluated"
] = True

metadata[
    "final_test_f1"
] = float(f1)

metadata[
    "final_test_roc_auc"
] = float(roc_auc)

metadata[
    "final_test_pr_auc"
] = float(pr_auc)

with open(
    META_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


print("\nMatriz de confusion:")

print(
    f"""
                 Pred Normal   Pred Premium
Real Normal      {tn:11d}   {fp:12d}
Real Premium     {fn:11d}   {tp:12d}
"""
)

print(
    "TEST FINAL EVALUADO."
)

print(
    "A partir de ahora no usaremos "
    "este test para seleccionar nuevos modelos."
)
