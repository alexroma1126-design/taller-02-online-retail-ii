from pathlib import Path
import sys

import pandas as pd


print("=" * 78)
print("AUDITORIA FINAL - MODELOS MEJORADOS")
print("=" * 78)


checks = []


def check(name, condition, detail=""):
    ok = bool(condition)

    checks.append(
        {
            "Check": name,
            "OK": ok,
            "Detail": detail,
        }
    )

    status = "OK" if ok else "FAIL"

    print(
        f"{status:4s} | {name}"
        + (
            f" | {detail}"
            if detail
            else ""
        )
    )


# ============================================================
# 1. ARCHIVOS
# ============================================================

required_files = [
    Path(
        "outputs/champion/"
        "classification_champion_metrics.csv"
    ),
    Path(
        "outputs/champion/"
        "classification_v1_vs_v2_comparison.csv"
    ),
    Path(
        "outputs/champion/"
        "clustering_champion_comparison.csv"
    ),
    Path(
        "outputs/champion/"
        "clustering_champion_profiles.csv"
    ),
    Path(
        "outputs/champion/"
        "clustering_champion_stability.csv"
    ),
    Path(
        "outputs/champion/"
        "forecasting_champion_metrics.csv"
    ),
    Path(
        "outputs/champion/"
        "forecasting_champion_predictions.csv"
    ),
    Path(
        "outputs/champion/"
        "FINAL_MODEL_SUMMARY.csv"
    ),
]

for file in required_files:
    check(
        f"Existe {file}",
        file.exists(),
    )


# ============================================================
# 2. CLASIFICACION
# ============================================================

classification = pd.read_csv(
    "outputs/champion/"
    "classification_champion_metrics.csv"
)

rf = classification[
    classification["Model"]
    == "Random Forest"
].iloc[0]

check(
    "Clasificacion final = Random Forest",
    rf["Model"] == "Random Forest",
)

check(
    "F1 clasificacion >= 0.70",
    float(rf["F1"]) >= 0.70,
    f"F1={float(rf['F1']):.4f}",
)

check(
    "ROC-AUC clasificacion >= 0.87",
    float(rf["ROC_AUC"]) >= 0.87,
    f"AUC={float(rf['ROC_AUC']):.4f}",
)


# ============================================================
# 3. CLUSTERING
# ============================================================

clustering = pd.read_csv(
    "outputs/champion/"
    "clustering_champion_comparison.csv"
)

km_v2 = clustering[
    clustering["Version"]
    == "V2 K-Means"
].iloc[0]

km_v1 = clustering[
    clustering["Version"]
    == "V1 K-Means"
].iloc[0]

check(
    "K-Means V2 supera Silhouette V1",
    float(km_v2["Silhouette"])
    > float(km_v1["Silhouette"]),
    (
        f"{float(km_v1['Silhouette']):.4f}"
        f" -> "
        f"{float(km_v2['Silhouette']):.4f}"
    ),
)

check(
    "Silhouette K-Means V2 >= 0.43",
    float(km_v2["Silhouette"]) >= 0.43,
    f"{float(km_v2['Silhouette']):.4f}",
)

stability = pd.read_csv(
    "outputs/champion/"
    "clustering_champion_stability.csv"
)

mean_ari = float(
    stability[
        "AdjustedRandIndex"
    ].mean()
)

check(
    "ARI promedio >= 0.99",
    mean_ari >= 0.99,
    f"ARI={mean_ari:.4f}",
)


# ============================================================
# 4. FORECASTING
# ============================================================

forecast = pd.read_csv(
    "outputs/champion/"
    "forecasting_champion_metrics.csv"
)

champion = forecast[
    forecast["Model"]
    == "Extra Trees + Calendar"
].iloc[0]

mae = float(
    champion["MAE"]
)

r2 = float(
    champion["R2"]
)

smape = float(
    champion["sMAPE"]
)

check(
    "Forecast champion identificado",
    champion["Model"]
    == "Extra Trees + Calendar",
)

check(
    "MAE forecasting < 9600",
    mae < 9600,
    f"MAE={mae:.2f}",
)

check(
    "R2 forecasting > 0.52",
    r2 > 0.52,
    f"R2={r2:.4f}",
)

check(
    "sMAPE forecasting < 32",
    smape < 32,
    f"sMAPE={smape:.2f}",
)


# ============================================================
# 5. RESUMEN MAESTRO
# ============================================================

summary = pd.read_csv(
    "outputs/champion/"
    "FINAL_MODEL_SUMMARY.csv"
)

check(
    "Resumen contiene tres tareas",
    len(summary) == 3,
    f"Filas={len(summary)}",
)

required_tasks = {
    "Classification",
    "Clustering",
    "Sales forecasting",
}

check(
    "Resumen contiene todas las tareas",
    set(summary["Task"])
    == required_tasks,
)


# ============================================================
# 6. README
# ============================================================

readme = Path(
    "README.md"
).read_text(
    encoding="utf-8"
)

for phrase in [
    "Optimizacion y seleccion final de modelos",
    "Silhouette = 0.4386",
    "MAE = 9,576.10",
    "F1 = 0.7010",
]:

    check(
        f"README contiene: {phrase}",
        phrase in readme,
    )


# ============================================================
# RESULTADO
# ============================================================

results = pd.DataFrame(
    checks
)

results.to_csv(
    "outputs/champion/"
    "FINAL_VALIDATION.csv",
    index=False,
)

failed = results[
    ~results["OK"]
]

print("\n" + "=" * 78)

if len(failed) == 0:

    print(
        f"AUDITORIA SUPERADA: "
        f"{len(results)}/{len(results)} verificaciones OK"
    )

    sys.exit(0)

else:

    print(
        f"AUDITORIA FALLIDA: "
        f"{len(failed)} verificaciones con error"
    )

    print(
        failed.to_string(
            index=False
        )
    )

    sys.exit(1)
