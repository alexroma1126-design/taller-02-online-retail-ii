from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_predict,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# CLASIFICACION DE CLIENTES V2
# SELECCION DE MODELO SIN TOCAR EL TEST FINAL
# ============================================================

INPUT_FILE = Path(
    "outputs/online_retail_clean.csv"
)

OUTPUT_DIR = Path(
    "outputs/classification_v2"
)

MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_STATE = 42

print("=" * 82)
print("CLASIFICACION DE CLIENTES V2 - SELECCION")
print("=" * 82)


# ============================================================
# 1. DATOS
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["InvoiceDate"],
)

print(
    f"\nTransacciones: "
    f"{len(df):,}"
)

print(
    f"Clientes totales: "
    f"{df['CustomerID'].nunique():,}"
)


# ============================================================
# 2. MISMO CORTE TEMPORAL DE V1
# ============================================================

cutoff = pd.Timestamp(
    "2010-12-01"
)

history = df[
    df["InvoiceDate"] < cutoff
].copy()

future = df[
    df["InvoiceDate"] >= cutoff
].copy()


# ============================================================
# 3. MISMAS FEATURES DE V1
# ============================================================

customer_features = (
    history
    .groupby("CustomerID")
    .agg(
        PastSales=(
            "Sales",
            "sum",
        ),
        Frequency=(
            "Invoice",
            "nunique",
        ),
        TotalQuantity=(
            "Quantity",
            "sum",
        ),
        UniqueProducts=(
            "StockCode",
            "nunique",
        ),
        ActiveDays=(
            "InvoiceDate",
            lambda x:
                x.dt.date.nunique(),
        ),
        AvgUnitPrice=(
            "Price",
            "mean",
        ),
        FirstPurchase=(
            "InvoiceDate",
            "min",
        ),
        LastPurchase=(
            "InvoiceDate",
            "max",
        ),
    )
    .reset_index()
)

customer_features[
    "AvgItemsPerInvoice"
] = (
    customer_features[
        "TotalQuantity"
    ]
    /
    customer_features[
        "Frequency"
    ]
)

customer_features[
    "RecencyDays"
] = (
    cutoff -
    customer_features[
        "LastPurchase"
    ]
).dt.days

customer_features[
    "TenureDays"
] = (
    customer_features[
        "LastPurchase"
    ]
    -
    customer_features[
        "FirstPurchase"
    ]
).dt.days + 1

customer_features = (
    customer_features
    .drop(
        columns=[
            "FirstPurchase",
            "LastPurchase",
        ]
    )
)


# ============================================================
# 4. MISMA DEFINICION PREMIUM DE V1
# ============================================================

future_behavior = (
    future
    .groupby("CustomerID")
    .agg(
        FutureSales=(
            "Sales",
            "sum",
        ),
        FutureInvoices=(
            "Invoice",
            "nunique",
        ),
        FutureQuantity=(
            "Quantity",
            "sum",
        ),
    )
    .reset_index()
)

customers = (
    customer_features
    .merge(
        future_behavior,
        on="CustomerID",
        how="inner",
    )
)

premium_threshold = (
    customers[
        "FutureSales"
    ]
    .quantile(0.75)
)

customers[
    "Premium"
] = (
    customers[
        "FutureSales"
    ] >= premium_threshold
).astype(int)

print(
    f"\nClientes utilizados: "
    f"{len(customers):,}"
)

print(
    f"Umbral Premium V1: "
    f"GBP {premium_threshold:,.2f}"
)

print(
    f"Premium: "
    f"{customers['Premium'].sum():,}"
)

print(
    f"Normal: "
    f"{(1-customers['Premium']).sum():,}"
)


# ============================================================
# 5. MISMAS 9 FEATURES
# ============================================================

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

X = customers[
    feature_columns
].copy()

y = customers[
    "Premium"
].copy()


# ============================================================
# 6. MISMO SPLIT DE V1
# ============================================================

(
    X_train,
    X_test,
    y_train,
    y_test,
) = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nSplit preservado:")

print(
    f"Train para seleccion: "
    f"{len(X_train):,}"
)

print(
    f"Test FINAL reservado:  "
    f"{len(X_test):,}"
)

print(
    "\nIMPORTANTE: "
    "el test final NO sera evaluado "
    "en este script."
)


# ============================================================
# 7. GUARDAR IDENTIDAD DEL SPLIT
# ============================================================

split_membership = pd.DataFrame(
    {
        "CustomerID":
            customers["CustomerID"],

        "Split":
            np.where(
                customers.index.isin(
                    X_test.index
                ),
                "test",
                "train",
            ),
    }
)

split_membership.to_csv(
    OUTPUT_DIR /
    "classification_split.csv",
    index=False,
)


# ============================================================
# 8. VALIDACION CRUZADA
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# 9. FUNCIONES
# ============================================================

def metrics_from_probabilities(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "Accuracy":
            accuracy_score(
                y_true,
                predictions,
            ),

        "Precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "Recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "F1":
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "ROC_AUC":
            roc_auc_score(
                y_true,
                probabilities,
            ),

        "PR_AUC":
            average_precision_score(
                y_true,
                probabilities,
            ),
    }


def optimize_threshold(
    y_true,
    probabilities,
):

    rows = []

    for threshold in np.arange(
        0.15,
        0.851,
        0.01,
    ):

        metrics = (
            metrics_from_probabilities(
                y_true,
                probabilities,
                threshold,
            )
        )

        rows.append(
            {
                "Threshold":
                    threshold,

                **metrics,
            }
        )

    results = pd.DataFrame(
        rows
    )

    results[
        "DistanceTo05"
    ] = (
        results[
            "Threshold"
        ] - 0.50
    ).abs()

    best = (
        results
        .sort_values(
            [
                "F1",
                "PR_AUC",
                "Accuracy",
                "DistanceTo05",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .iloc[0]
    )

    return (
        float(
            best["Threshold"]
        ),
        results,
    )


# ============================================================
# 10. V1 RANDOM FOREST COMO REFERENCIA INTERNA
# ============================================================

print("\n" + "=" * 82)
print("REFERENCIA V1 EN VALIDACION CRUZADA")
print("=" * 82)

v1_rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

v1_oof_prob = (
    cross_val_predict(
        v1_rf,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
)

v1_default = (
    metrics_from_probabilities(
        y_train,
        v1_oof_prob,
        threshold=0.50,
    )
)

(
    v1_best_threshold,
    v1_threshold_table,
) = optimize_threshold(
    y_train,
    v1_oof_prob,
)

v1_optimized = (
    metrics_from_probabilities(
        y_train,
        v1_oof_prob,
        threshold=v1_best_threshold,
    )
)

print(
    f"\nV1 RF threshold 0.50:"
)

print(
    f"F1={v1_default['F1']:.4f} | "
    f"Precision={v1_default['Precision']:.4f} | "
    f"Recall={v1_default['Recall']:.4f} | "
    f"AUC={v1_default['ROC_AUC']:.4f}"
)

print(
    f"\nV1 RF threshold optimizado "
    f"{v1_best_threshold:.2f}:"
)

print(
    f"F1={v1_optimized['F1']:.4f} | "
    f"Precision={v1_optimized['Precision']:.4f} | "
    f"Recall={v1_optimized['Recall']:.4f} | "
    f"AUC={v1_optimized['ROC_AUC']:.4f}"
)

v1_threshold_table.to_csv(
    OUTPUT_DIR /
    "threshold_search_v1_rf.csv",
    index=False,
)


# ============================================================
# 11. MODELOS CANDIDATOS
# ============================================================

models = {
    "Logistic Regression": {
        "estimator":
            Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=5000,
                            class_weight="balanced",
                            random_state=
                                RANDOM_STATE,
                        ),
                    ),
                ]
            ),

        "params": {
            "model__C":
                np.logspace(
                    -3,
                    2,
                    16,
                ),

            "model__solver": [
                "lbfgs",
                "liblinear",
            ],
        },

        "n_iter":
            20,
    },


    "Random Forest": {
        "estimator":
            RandomForestClassifier(
                random_state=
                    RANDOM_STATE,
                n_jobs=-1,
            ),

        "params": {
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
                8,
                12,
            ],

            "min_samples_leaf": [
                1,
                2,
                4,
                6,
                8,
            ],

            "max_features": [
                "sqrt",
                0.5,
                0.7,
                1.0,
            ],

            "class_weight": [
                "balanced",
                "balanced_subsample",
            ],
        },

        "n_iter":
            20,
    },


    "Extra Trees": {
        "estimator":
            ExtraTreesClassifier(
                random_state=
                    RANDOM_STATE,
                n_jobs=-1,
            ),

        "params": {
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
                8,
                12,
            ],

            "min_samples_leaf": [
                1,
                2,
                4,
                6,
                8,
            ],

            "max_features": [
                "sqrt",
                0.5,
                0.7,
                1.0,
            ],

            "class_weight": [
                "balanced",
                "balanced_subsample",
            ],
        },

        "n_iter":
            20,
    },


    "HistGradientBoosting": {
        "estimator":
            HistGradientBoostingClassifier(
                random_state=
                    RANDOM_STATE,
                class_weight=
                    "balanced",
            ),

        "params": {
            "learning_rate": [
                0.02,
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
                10,
                20,
                30,
                40,
            ],

            "l2_regularization": [
                0.0,
                0.1,
                1.0,
                5.0,
            ],
        },

        "n_iter":
            20,
    },
}


# ============================================================
# 12. BUSQUEDA SOLO SOBRE TRAIN
# ============================================================

selection_rows = []

best_estimators = {}

best_parameters = {}

best_thresholds = {}


for name, config in models.items():

    print(
        "\n" +
        "=" * 82
    )

    print(
        f"TUNING: {name}"
    )

    print(
        "=" * 82
    )

    search = RandomizedSearchCV(
        estimator=
            config["estimator"],

        param_distributions=
            config["params"],

        n_iter=
            config["n_iter"],

        scoring="f1",

        cv=cv,

        random_state=
            RANDOM_STATE,

        n_jobs=-1,

        verbose=1,

        refit=True,
    )

    search.fit(
        X_train,
        y_train,
    )

    best_model = (
        search
        .best_estimator_
    )

    best_estimators[
        name
    ] = best_model

    best_parameters[
        name
    ] = search.best_params_


    # --------------------------------------------------------
    # OOF probabilities: nunca usamos X_test
    # --------------------------------------------------------

    oof_prob = (
        cross_val_predict(
            clone(
                best_model
            ),
            X_train,
            y_train,
            cv=cv,
            method=
                "predict_proba",
            n_jobs=1,
        )[:, 1]
    )


    # --------------------------------------------------------
    # Threshold elegido con OOF train
    # --------------------------------------------------------

    (
        threshold,
        threshold_table,
    ) = optimize_threshold(
        y_train,
        oof_prob,
    )

    best_thresholds[
        name
    ] = threshold

    metrics = (
        metrics_from_probabilities(
            y_train,
            oof_prob,
            threshold,
        )
    )

    safe_name = (
        name.lower()
        .replace(
            " ",
            "_",
        )
    )

    threshold_table.to_csv(
        OUTPUT_DIR /
        (
            "threshold_search_" +
            safe_name +
            ".csv"
        ),
        index=False,
    )

    selection_rows.append(
        {
            "Model":
                name,

            "SearchCV_F1":
                search.best_score_,

            "Threshold":
                threshold,

            **metrics,
        }
    )

    print(
        "\nMejores parametros:"
    )

    print(
        search.best_params_
    )

    print(
        f"\nSearchCV F1: "
        f"{search.best_score_:.4f}"
    )

    print(
        f"Threshold OOF: "
        f"{threshold:.2f}"
    )

    print(
        f"OOF Accuracy:  "
        f"{metrics['Accuracy']:.4f}"
    )

    print(
        f"OOF Precision: "
        f"{metrics['Precision']:.4f}"
    )

    print(
        f"OOF Recall:    "
        f"{metrics['Recall']:.4f}"
    )

    print(
        f"OOF F1:        "
        f"{metrics['F1']:.4f}"
    )

    print(
        f"OOF ROC-AUC:   "
        f"{metrics['ROC_AUC']:.4f}"
    )

    print(
        f"OOF PR-AUC:    "
        f"{metrics['PR_AUC']:.4f}"
    )


# ============================================================
# 13. COMPARACION
# ============================================================

selection_df = (
    pd.DataFrame(
        selection_rows
    )
    .sort_values(
        [
            "F1",
            "PR_AUC",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)

print("\n" + "=" * 82)
print("SELECCION V2 - SOLO DATOS DE TRAIN")
print("=" * 82)

print(
    selection_df
    .round(
        {
            "SearchCV_F1": 4,
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
# 14. CAMPEON ELEGIDO SIN VER TEST
# ============================================================

champion_name = (
    selection_df
    .iloc[0][
        "Model"
    ]
)

champion_threshold = (
    best_thresholds[
        champion_name
    ]
)

champion_model = (
    best_estimators[
        champion_name
    ]
)

print("\n" + "=" * 82)
print("CAMPEON SELECCIONADO SIN VER TEST")
print("=" * 82)

print(
    f"Modelo: "
    f"{champion_name}"
)

print(
    f"Threshold: "
    f"{champion_threshold:.2f}"
)

print(
    f"F1 OOF: "
    f"{selection_df.iloc[0]['F1']:.4f}"
)

print(
    f"ROC-AUC OOF: "
    f"{selection_df.iloc[0]['ROC_AUC']:.4f}"
)

print(
    f"PR-AUC OOF: "
    f"{selection_df.iloc[0]['PR_AUC']:.4f}"
)


# ============================================================
# 15. COMPARACION CON V1 OOF
# ============================================================

champion_oof_f1 = (
    selection_df
    .iloc[0][
        "F1"
    ]
)

improvement_vs_v1_default = (
    (
        champion_oof_f1 -
        v1_default["F1"]
    )
    /
    v1_default["F1"]
    *
    100
)

improvement_vs_v1_threshold = (
    (
        champion_oof_f1 -
        v1_optimized["F1"]
    )
    /
    v1_optimized["F1"]
    *
    100
)

print(
    "\nMejora F1 OOF "
    "vs V1 threshold 0.50: "
    f"{improvement_vs_v1_default:.2f}%"
)

print(
    "Mejora F1 OOF "
    "vs V1 con threshold optimizado: "
    f"{improvement_vs_v1_threshold:.2f}%"
)


# ============================================================
# 16. GUARDAR
# ============================================================

selection_df.to_csv(
    OUTPUT_DIR /
    "model_selection_oof.csv",
    index=False,
)

customers.to_csv(
    OUTPUT_DIR /
    "customer_classification_dataset.csv",
    index=False,
)

with open(
    OUTPUT_DIR /
    "best_parameters.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        best_parameters,
        f,
        indent=2,
        default=str,
    )

with open(
    OUTPUT_DIR /
    "best_thresholds.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        best_thresholds,
        f,
        indent=2,
    )

metadata = {
    "cutoff":
        str(
            cutoff.date()
        ),

    "premium_threshold":
        float(
            premium_threshold
        ),

    "train_size":
        int(
            len(X_train)
        ),

    "test_size":
        int(
            len(X_test)
        ),

    "champion_model":
        champion_name,

    "champion_threshold":
        float(
            champion_threshold
        ),

    "champion_oof_f1":
        float(
            champion_oof_f1
        ),

    "test_evaluated":
        False,
}

with open(
    OUTPUT_DIR /
    "selection_metadata.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


# El modelo ya esta entrenado exclusivamente
# sobre X_train. El test sigue sin tocarse.

joblib.dump(
    champion_model,
    MODEL_DIR /
    "classification_v2_candidate.joblib",
)


print("\nArchivos generados:")

for file in sorted(
    OUTPUT_DIR.glob("*")
):

    print(
        f"  {file}"
    )


print(
    "\nSELECCION V2 COMPLETADA."
)

print(
    "EL TEST FINAL DE 546 CLIENTES "
    "SIGUE SIN EVALUARSE."
)
