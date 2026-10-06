from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.decomposition import PCA
import joblib


# ============================================================
# TALLER 02 - MAESTRIA EN CIENCIA DE DATOS
# PARTE 2: SEGMENTACION DE CLIENTES
# K-MEANS Y MEAN SHIFT
# ============================================================

INPUT_FILE = Path("outputs/online_retail_clean.csv")

OUTPUT_DIR = Path("outputs/clustering")
FIG_DIR = Path("figures/clustering")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("=" * 78)
print("TALLER 02 - SEGMENTACION DE CLIENTES")
print("=" * 78)


# ------------------------------------------------------------
# 1. CARGAR DATOS
# ------------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["InvoiceDate"]
)

print(f"\nTransacciones: {len(df):,}")
print(f"Clientes:     {df['CustomerID'].nunique():,}")


# ------------------------------------------------------------
# 2. CONSTRUIR DATASET A NIVEL CLIENTE
# ------------------------------------------------------------
# Usaremos variables tipo RFM ampliadas:
#
# Recency   = dias desde la ultima compra
# Frequency = numero de facturas
# Monetary  = gasto total
#
# Y agregamos:
# Quantity, UniqueProducts y AvgOrderValue
# ------------------------------------------------------------

reference_date = (
    df["InvoiceDate"].max() +
    pd.Timedelta(days=1)
)

customers = (
    df.groupby("CustomerID")
    .agg(
        Monetary=("Sales", "sum"),
        Frequency=("Invoice", "nunique"),
        Quantity=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        LastPurchase=("InvoiceDate", "max"),
    )
    .reset_index()
)

customers["Recency"] = (
    reference_date -
    customers["LastPurchase"]
).dt.days

customers["AvgOrderValue"] = (
    customers["Monetary"] /
    customers["Frequency"]
)

customers = customers.drop(
    columns=["LastPurchase"]
)


# ------------------------------------------------------------
# 3. VARIABLES PARA CLUSTERING
# ------------------------------------------------------------

features = [
    "Recency",
    "Frequency",
    "Monetary",
    "Quantity",
    "UniqueProducts",
    "AvgOrderValue",
]

print("\nVariables de segmentacion:")

for feature in features:
    print(f"  - {feature}")


# ------------------------------------------------------------
# 4. REVISAR DISTRIBUCIONES
# ------------------------------------------------------------

print("\nResumen de clientes:")

print(
    customers[features]
    .describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.95,
            0.99
        ]
    )
    .round(2)
    .to_string()
)


# ------------------------------------------------------------
# 5. TRANSFORMACION LOGARITMICA
# ------------------------------------------------------------
# Las variables monetarias y de frecuencia son muy sesgadas.
# log1p reduce el efecto de valores extremos sin eliminarlos.
# ------------------------------------------------------------

cluster_data = customers[
    features
].copy()

for column in features:
    cluster_data[column] = np.log1p(
        cluster_data[column]
    )


# ------------------------------------------------------------
# 6. ESCALADO
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    cluster_data
)

joblib.dump(
    scaler,
    MODEL_DIR / "customer_clustering_scaler.joblib"
)


# ------------------------------------------------------------
# 7. SELECCION DE K PARA K-MEANS
# ------------------------------------------------------------

print("\n" + "=" * 78)
print("SELECCION DE K PARA K-MEANS")
print("=" * 78)

k_results = []

for k in range(2, 9):

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=20
    )

    labels = model.fit_predict(
        X_scaled
    )

    silhouette = silhouette_score(
        X_scaled,
        labels
    )

    calinski = calinski_harabasz_score(
        X_scaled,
        labels
    )

    davies = davies_bouldin_score(
        X_scaled,
        labels
    )

    k_results.append(
        {
            "K": k,
            "Silhouette": silhouette,
            "CalinskiHarabasz": calinski,
            "DaviesBouldin": davies,
            "Inertia": model.inertia_,
        }
    )

    print(
        f"K={k}: "
        f"Silhouette={silhouette:.4f} | "
        f"Calinski={calinski:.2f} | "
        f"Davies-Bouldin={davies:.4f} | "
        f"Inertia={model.inertia_:.2f}"
    )


k_results_df = pd.DataFrame(
    k_results
)

k_results_df.to_csv(
    OUTPUT_DIR /
    "kmeans_k_selection.csv",
    index=False
)


# ------------------------------------------------------------
# 8. ELEGIR EL MEJOR K
# ------------------------------------------------------------

best_k = int(
    k_results_df.loc[
        k_results_df["Silhouette"].idxmax(),
        "K"
    ]
)

print(
    f"\nMejor K segun Silhouette: {best_k}"
)


# ------------------------------------------------------------
# 9. ENTRENAR K-MEANS FINAL
# ------------------------------------------------------------

kmeans = KMeans(
    n_clusters=best_k,
    random_state=RANDOM_STATE,
    n_init=30
)

customers["KMeansCluster"] = (
    kmeans.fit_predict(
        X_scaled
    )
)

joblib.dump(
    kmeans,
    MODEL_DIR /
    "kmeans_customer_segmentation.joblib"
)


# ------------------------------------------------------------
# 10. PERFIL DE CLUSTERS K-MEANS
# ------------------------------------------------------------

kmeans_profile = (
    customers.groupby(
        "KMeansCluster"
    )
    .agg(
        Customers=("CustomerID", "count"),
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean"),
        Quantity=("Quantity", "mean"),
        UniqueProducts=("UniqueProducts", "mean"),
        AvgOrderValue=("AvgOrderValue", "mean"),
    )
    .reset_index()
)

kmeans_profile[
    "Percentage"
] = (
    kmeans_profile["Customers"] /
    len(customers) *
    100
)

kmeans_profile.to_csv(
    OUTPUT_DIR /
    "kmeans_cluster_profiles.csv",
    index=False
)

print("\n" + "=" * 78)
print("PERFILES K-MEANS")
print("=" * 78)

print(
    kmeans_profile
    .round(2)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 11. GRAFICO SELECCION K
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    k_results_df["K"],
    k_results_df["Silhouette"],
    marker="o"
)

plt.xlabel("Numero de clusters K")
plt.ylabel("Silhouette Score")
plt.title("Seleccion de K mediante Silhouette")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "01_kmeans_silhouette_by_k.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# 12. METODO DEL CODO
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    k_results_df["K"],
    k_results_df["Inertia"],
    marker="o"
)

plt.xlabel("Numero de clusters K")
plt.ylabel("Inercia")
plt.title("Metodo del codo - K-means")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "02_kmeans_elbow.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# 13. MEAN SHIFT - ESTIMAR BANDWIDTH
# ------------------------------------------------------------

print("\n" + "=" * 78)
print("MEAN SHIFT")
print("=" * 78)

sample_size = min(
    3000,
    len(X_scaled)
)

bandwidth = estimate_bandwidth(
    X_scaled,
    quantile=0.20,
    n_samples=sample_size,
    random_state=RANDOM_STATE
)

print(
    f"\nBandwidth estimado: "
    f"{bandwidth:.4f}"
)

if bandwidth <= 0:
    bandwidth = 1.0
    print(
        "Bandwidth invalido; "
        "se utilizara 1.0."
    )


# ------------------------------------------------------------
# 14. ENTRENAR MEAN SHIFT
# ------------------------------------------------------------

meanshift = MeanShift(
    bandwidth=bandwidth,
    bin_seeding=True,
    cluster_all=True
)

customers[
    "MeanShiftCluster"
] = meanshift.fit_predict(
    X_scaled
)

joblib.dump(
    meanshift,
    MODEL_DIR /
    "meanshift_customer_segmentation.joblib"
)

n_meanshift_clusters = (
    customers[
        "MeanShiftCluster"
    ]
    .nunique()
)

print(
    f"Clusters encontrados por Mean Shift: "
    f"{n_meanshift_clusters}"
)


# ------------------------------------------------------------
# 15. PERFIL MEAN SHIFT
# ------------------------------------------------------------

meanshift_profile = (
    customers.groupby(
        "MeanShiftCluster"
    )
    .agg(
        Customers=("CustomerID", "count"),
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean"),
        Quantity=("Quantity", "mean"),
        UniqueProducts=("UniqueProducts", "mean"),
        AvgOrderValue=("AvgOrderValue", "mean"),
    )
    .reset_index()
)

meanshift_profile[
    "Percentage"
] = (
    meanshift_profile["Customers"] /
    len(customers) *
    100
)

meanshift_profile.to_csv(
    OUTPUT_DIR /
    "meanshift_cluster_profiles.csv",
    index=False
)

print("\n" + "=" * 78)
print("PERFILES MEAN SHIFT")
print("=" * 78)

print(
    meanshift_profile
    .round(2)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 16. METRICAS COMPARATIVAS
# ------------------------------------------------------------

kmeans_labels = customers[
    "KMeansCluster"
].to_numpy()

meanshift_labels = customers[
    "MeanShiftCluster"
].to_numpy()


def clustering_metrics(
    method,
    labels
):

    n_clusters = len(
        np.unique(labels)
    )

    result = {
        "Method": method,
        "Clusters": n_clusters,
        "Silhouette": np.nan,
        "CalinskiHarabasz": np.nan,
        "DaviesBouldin": np.nan,
    }

    if (
        n_clusters > 1 and
        n_clusters < len(labels)
    ):

        result[
            "Silhouette"
        ] = silhouette_score(
            X_scaled,
            labels
        )

        result[
            "CalinskiHarabasz"
        ] = calinski_harabasz_score(
            X_scaled,
            labels
        )

        result[
            "DaviesBouldin"
        ] = davies_bouldin_score(
            X_scaled,
            labels
        )

    return result


comparison = pd.DataFrame(
    [
        clustering_metrics(
            "K-Means",
            kmeans_labels
        ),
        clustering_metrics(
            "Mean Shift",
            meanshift_labels
        )
    ]
)

comparison.to_csv(
    OUTPUT_DIR /
    "clustering_comparison.csv",
    index=False
)

print("\n" + "=" * 78)
print("COMPARACION K-MEANS VS MEAN SHIFT")
print("=" * 78)

print(
    comparison
    .round(4)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 17. PCA SOLO PARA VISUALIZACION
# ------------------------------------------------------------

pca = PCA(
    n_components=2,
    random_state=RANDOM_STATE
)

X_pca = pca.fit_transform(
    X_scaled
)

customers["PC1"] = X_pca[:, 0]
customers["PC2"] = X_pca[:, 1]

print(
    "\nVarianza explicada por PCA:"
)

print(
    f"  PC1: "
    f"{pca.explained_variance_ratio_[0] * 100:.2f}%"
)

print(
    f"  PC2: "
    f"{pca.explained_variance_ratio_[1] * 100:.2f}%"
)

print(
    f"  Total: "
    f"{pca.explained_variance_ratio_.sum() * 100:.2f}%"
)


# ------------------------------------------------------------
# 18. GRAFICO PCA K-MEANS
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    customers["PC1"],
    customers["PC2"],
    c=customers["KMeansCluster"],
    s=18,
    alpha=0.65
)

plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.title(
    f"Segmentacion K-means "
    f"(K={best_k})"
)

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "03_kmeans_pca.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# 19. GRAFICO PCA MEAN SHIFT
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    customers["PC1"],
    customers["PC2"],
    c=customers["MeanShiftCluster"],
    s=18,
    alpha=0.65
)

plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.title(
    "Segmentacion Mean Shift"
)

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "04_meanshift_pca.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# 20. TAMAÑO DE CLUSTERS K-MEANS
# ------------------------------------------------------------

k_counts = (
    customers[
        "KMeansCluster"
    ]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(9, 6))

plt.bar(
    k_counts.index.astype(str),
    k_counts.values
)

plt.xlabel("Cluster")
plt.ylabel("Numero de clientes")
plt.title("Tamaño de clusters K-means")
plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "05_kmeans_cluster_sizes.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# 21. TAMAÑO DE CLUSTERS MEAN SHIFT
# ------------------------------------------------------------

ms_counts = (
    customers[
        "MeanShiftCluster"
    ]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(10, 6))

plt.bar(
    ms_counts.index.astype(str),
    ms_counts.values
)

plt.xlabel("Cluster")
plt.ylabel("Numero de clientes")
plt.title("Tamaño de clusters Mean Shift")
plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "06_meanshift_cluster_sizes.png",
    dpi=160
)

plt.close()


# ------------------------------------------------------------
# 22. GUARDAR ASIGNACIONES
# ------------------------------------------------------------

customers.to_csv(
    OUTPUT_DIR /
    "customer_segments.csv",
    index=False
)


# ------------------------------------------------------------
# 23. RESUMEN
# ------------------------------------------------------------

best_k_silhouette = (
    comparison.loc[
        comparison["Method"] == "K-Means",
        "Silhouette"
    ]
    .iloc[0]
)

meanshift_silhouette = (
    comparison.loc[
        comparison["Method"] == "Mean Shift",
        "Silhouette"
    ]
    .iloc[0]
)

summary_lines = [
    "TALLER 02 - SEGMENTACION DE CLIENTES",
    "=" * 65,
    "",
    f"Clientes analizados: {len(customers):,}",
    "",
    "Variables:",
    *[
        f"- {x}"
        for x in features
    ],
    "",
    "K-MEANS",
    f"Mejor K: {best_k}",
    (
        f"Silhouette: "
        f"{best_k_silhouette:.4f}"
    ),
    "",
    "MEAN SHIFT",
    (
        f"Bandwidth: "
        f"{bandwidth:.4f}"
    ),
    (
        f"Clusters encontrados: "
        f"{n_meanshift_clusters}"
    ),
    (
        f"Silhouette: "
        f"{meanshift_silhouette:.4f}"
        if not np.isnan(
            meanshift_silhouette
        )
        else "Silhouette: no definido"
    ),
    "",
    "COMPARACION",
    comparison
    .round(4)
    .to_string(index=False)
]

(
    OUTPUT_DIR /
    "CLUSTERING_SUMMARY.txt"
).write_text(
    "\n".join(summary_lines),
    encoding="utf-8"
)


print("\n" + "=" * 78)
print("ARCHIVOS GENERADOS")
print("=" * 78)

print("\nTablas:")

for file in sorted(
    OUTPUT_DIR.glob("*")
):
    print(f"  {file}")

print("\nFiguras:")

for file in sorted(
    FIG_DIR.glob("*.png")
):
    print(f"  {file}")

print("\nPARTE 2 COMPLETADA CORRECTAMENTE.")
