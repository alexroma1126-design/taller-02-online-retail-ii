from pathlib import Path
import pandas as pd
import sys


print("=" * 78)
print("AUDITORIA FINAL - TALLER 02")
print("=" * 78)

errors = []
checks = []


def check(condition, message):
    if condition:
        print(f"[OK]   {message}")
        checks.append(message)
    else:
        print(f"[FAIL] {message}")
        errors.append(message)


# ============================================================
# 1. SCRIPTS
# ============================================================

required_scripts = [
    "src/01_prepare_data.py",
    "src/02_eda.py",
    "src/03_customer_classification.py",
    "src/04_customer_segmentation.py",
    "src/05_sales_prediction.py",
]

print("\n--- SCRIPTS ---")

for filename in required_scripts:
    check(
        Path(filename).exists(),
        filename
    )


# ============================================================
# 2. DATASET LIMPIO
# ============================================================

print("\n--- PREPARACION ---")

clean_path = Path(
    "outputs/online_retail_clean.csv"
)

check(
    clean_path.exists(),
    "Dataset limpio existe"
)

if clean_path.exists():

    df = pd.read_csv(
        clean_path,
        usecols=[
            "Invoice",
            "StockCode",
            "Quantity",
            "Price",
            "CustomerID",
            "Sales"
        ]
    )

    check(
        len(df) == 793609,
        f"Filas limpias = {len(df):,}"
    )

    check(
        df["CustomerID"].isna().sum() == 0,
        "No existen CustomerID faltantes"
    )

    check(
        (df["Quantity"] > 0).all(),
        "Todas las cantidades son positivas"
    )

    check(
        (df["Price"] > 0).all(),
        "Todos los precios son positivos"
    )


# ============================================================
# 3. EDA
# ============================================================

print("\n--- EDA ---")

eda_figures = list(
    Path("figures/eda").glob("*.png")
)

check(
    len(eda_figures) == 8,
    f"Figuras EDA = {len(eda_figures)}"
)

eda_tables = list(
    Path("outputs/eda").glob("*.csv")
)

check(
    len(eda_tables) == 7,
    f"Tablas EDA = {len(eda_tables)}"
)


# ============================================================
# 4. CLASIFICACION
# ============================================================

print("\n--- CLASIFICACION ---")

classification_path = Path(
    "outputs/classification/model_comparison.csv"
)

check(
    classification_path.exists(),
    "Comparacion de clasificadores existe"
)

if classification_path.exists():

    clf = pd.read_csv(
        classification_path
    )

    check(
        {"Random Forest",
         "Logistic Regression"}
        .issubset(set(clf["Model"])),
        "Modelos de clasificacion presentes"
    )

    rf = clf[
        clf["Model"] ==
        "Random Forest"
    ].iloc[0]

    check(
        rf["Accuracy"] > 0.7491,
        (
            "Random Forest supera "
            "baseline en accuracy"
        )
    )

    check(
        rf["ROC_AUC"] > 0.80,
        (
            f"Random Forest ROC-AUC "
            f"= {rf['ROC_AUC']:.4f}"
        )
    )


# ============================================================
# 5. CLUSTERING
# ============================================================

print("\n--- CLUSTERING ---")

clustering_path = Path(
    "outputs/clustering/clustering_comparison.csv"
)

check(
    clustering_path.exists(),
    "Comparacion clustering existe"
)

if clustering_path.exists():

    clustering = pd.read_csv(
        clustering_path
    )

    check(
        {"K-Means", "Mean Shift"}
        .issubset(set(
            clustering["Method"]
        )),
        "K-Means y Mean Shift presentes"
    )

    km = clustering[
        clustering["Method"] ==
        "K-Means"
    ].iloc[0]

    ms = clustering[
        clustering["Method"] ==
        "Mean Shift"
    ].iloc[0]

    check(
        km["Silhouette"] >
        ms["Silhouette"],
        (
            "K-Means supera Mean Shift "
            "en Silhouette"
        )
    )


# ============================================================
# 6. PREDICCION
# ============================================================

print("\n--- PREDICCION ---")

prediction_path = Path(
    "outputs/prediction/model_comparison.csv"
)

check(
    prediction_path.exists(),
    "Comparacion prediccion existe"
)

if prediction_path.exists():

    pred = pd.read_csv(
        prediction_path
    )

    expected_models = {
        "Baseline Lag-1",
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
    }

    check(
        expected_models.issubset(
            set(pred["Model"])
        ),
        "Todos los modelos predictivos presentes"
    )

    rf = pred[
        pred["Model"] ==
        "Random Forest"
    ].iloc[0]

    baseline = pred[
        pred["Model"] ==
        "Baseline Lag-1"
    ].iloc[0]

    improvement = (
        baseline["MAE"] -
        rf["MAE"]
    ) / baseline["MAE"] * 100

    check(
        rf["MAE"] <
        baseline["MAE"],
        (
            "Random Forest supera "
            "baseline en MAE"
        )
    )

    print(
        f"[INFO] Mejora RF vs baseline: "
        f"{improvement:.2f}%"
    )


# ============================================================
# 7. FIGURAS
# ============================================================

print("\n--- FIGURAS ---")

expected_counts = {
    "figures/eda": 8,
    "figures/classification": 4,
    "figures/clustering": 6,
    "figures/prediction": 4,
}

for folder, expected in expected_counts.items():

    count = len(
        list(
            Path(folder).glob("*.png")
        )
    )

    check(
        count == expected,
        f"{folder}: {count}/{expected}"
    )


# ============================================================
# RESULTADO
# ============================================================

print("\n" + "=" * 78)

if errors:

    print(
        f"AUDITORIA FALLIDA: "
        f"{len(errors)} problemas"
    )

    for error in errors:
        print(f"  - {error}")

    sys.exit(1)

else:

    print(
        f"AUDITORIA SUPERADA: "
        f"{len(checks)} verificaciones OK"
    )

    print(
        "El Taller 02 esta "
        "computacionalmente consistente."
    )

print("=" * 78)
