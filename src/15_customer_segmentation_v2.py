from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import (
    KMeans,
    MeanShift,
    estimate_bandwidth,
)
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
)
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# ============================================================
# CLUSTERING V2 - BUSQUEDA DE REPRESENTACION
# ============================================================

INPUT_FILE = Path(
    "outputs/online_retail_clean.csv"
)

V1_FILE = Path(
    "outputs/baseline_v1/clustering_v1.csv"
)

OUTPUT_DIR = Path(
    "outputs/clustering_v2"
)

FIG_DIR = Path(
    "figures/clustering_v2"
)

MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_STATE = 42
SIL_SAMPLE = 2500

print("=" * 82)
print("SEGMENTACION DE CLIENTES V2")
print("=" * 82)


# ============================================================
# 1. CARGAR DATOS
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
    f"Clientes: "
    f"{df['CustomerID'].nunique():,}"
)


# ============================================================
# 2. DATASET POR CLIENTE
# ============================================================

reference_date = (
    df["InvoiceDate"].max()
    + pd.Timedelta(days=1)
)

customers = (
    df.groupby("CustomerID")
    .agg(
        Monetary=(
            "Sales",
            "sum",
        ),
        Frequency=(
            "Invoice",
            "nunique",
        ),
        Quantity=(
            "Quantity",
            "sum",
        ),
        UniqueProducts=(
            "StockCode",
            "nunique",
        ),
        LastPurchase=(
            "InvoiceDate",
            "max",
        ),
    )
    .reset_index()
)

customers[
    "Recency"
] = (
    reference_date
    - customers["LastPurchase"]
).dt.days

customers[
    "AvgOrderValue"
] = (
    customers["Monetary"]
    / customers["Frequency"]
)

customers = customers.drop(
    columns=["LastPurchase"]
)

print(
    f"\nClientes modelables: "
    f"{len(customers):,}"
)


# ============================================================
# 3. CONJUNTOS DE FEATURES
# ============================================================
#
# A) RFM clasico
# B) RFM + diversidad de productos
# C) RFM + cantidad + productos
# D) Configuracion original V1
#
# AvgOrderValue no entra en las variantes A-C porque
# Monetary / Frequency ya contiene informacion relacionada.
# ============================================================

feature_sets = {
    "RFM": [
        "Recency",
        "Frequency",
        "Monetary",
    ],

    "RFM_Products": [
        "Recency",
        "Frequency",
        "Monetary",
        "UniqueProducts",
    ],

    "RFM_Extended": [
        "Recency",
        "Frequency",
        "Monetary",
        "Quantity",
        "UniqueProducts",
    ],

    "V1_Six_Features": [
        "Recency",
        "Frequency",
        "Monetary",
        "Quantity",
        "UniqueProducts",
        "AvgOrderValue",
    ],
}


scalers = {
    "StandardScaler":
        StandardScaler,

    "RobustScaler":
        RobustScaler,
}


# ============================================================
# 4. CORRELACIONES
# ============================================================

correlation_features = [
    "Recency",
    "Frequency",
    "Monetary",
    "Quantity",
    "UniqueProducts",
    "AvgOrderValue",
]

correlation = (
    customers[
        correlation_features
    ]
    .corr(
        method="spearman"
    )
)

correlation.to_csv(
    OUTPUT_DIR /
    "feature_spearman_correlation.csv"
)

print("\nCorrelacion Spearman:")

print(
    correlation
    .round(3)
    .to_string()
)


# ============================================================
# 5. HELPER PARA CONSTRUIR MATRIZ
# ============================================================

def build_matrix(
    feature_set_name,
    scaler_name,
):

    features = (
        feature_sets[
            feature_set_name
        ]
    )

    data = (
        customers[
            features
        ]
        .copy()
        .clip(lower=0)
    )

    # misma idea robusta de V1:
    # reducir asimetria sin eliminar outliers
    data = np.log1p(
        data
    )

    scaler = (
        scalers[
            scaler_name
        ]()
    )

    X = scaler.fit_transform(
        data
    )

    return (
        X,
        scaler,
        features,
    )


# ============================================================
# 6. BUSQUEDA K-MEANS
# ============================================================

print("\n" + "=" * 82)
print("BUSQUEDA K-MEANS")
print("=" * 82)

search_rows = []

for feature_set_name in feature_sets:

    for scaler_name in scalers:

        X, _, features = build_matrix(
            feature_set_name,
            scaler_name,
        )

        print(
            f"\n{feature_set_name} "
            f"+ {scaler_name}"
        )

        for k in range(
            2,
            9,
        ):

            model = KMeans(
                n_clusters=k,
                n_init=30,
                random_state=
                    RANDOM_STATE,
            )

            labels = (
                model.fit_predict(
                    X
                )
            )

            silhouette = (
                silhouette_score(
                    X,
                    labels,
                    sample_size=min(
                        SIL_SAMPLE,
                        len(X),
                    ),
                    random_state=
                        RANDOM_STATE,
                )
            )

            counts = np.bincount(
                labels
            )

            min_cluster_pct = (
                counts.min()
                / len(labels)
                * 100
            )

            max_cluster_pct = (
                counts.max()
                / len(labels)
                * 100
            )

            row = {
                "FeatureSet":
                    feature_set_name,

                "Scaler":
                    scaler_name,

                "K":
                    k,

                "SilhouetteSample":
                    silhouette,

                "CalinskiHarabasz":
                    calinski_harabasz_score(
                        X,
                        labels,
                    ),

                "DaviesBouldin":
                    davies_bouldin_score(
                        X,
                        labels,
                    ),

                "Inertia":
                    model.inertia_,

                "MinClusterPct":
                    min_cluster_pct,

                "MaxClusterPct":
                    max_cluster_pct,
            }

            search_rows.append(
                row
            )

            print(
                f"K={k} | "
                f"Sil={silhouette:.4f} | "
                f"DB={row['DaviesBouldin']:.4f} | "
                f"min={min_cluster_pct:.2f}% | "
                f"max={max_cluster_pct:.2f}%"
            )


search_df = pd.DataFrame(
    search_rows
)

search_df = (
    search_df
    .sort_values(
        "SilhouetteSample",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

search_df.to_csv(
    OUTPUT_DIR /
    "kmeans_full_search.csv",
    index=False,
)


# ============================================================
# 7. TOP CANDIDATOS CON SILHOUETTE EXACTO
# ============================================================
#
# La busqueda usa muestra para velocidad.
# Los 5 mejores se reevaluan sobre los 5,878 clientes.
# ============================================================

print("\n" + "=" * 82)
print("REEVALUACION EXACTA DE TOP 5")
print("=" * 82)

exact_rows = []

for _, row in (
    search_df
    .head(5)
    .iterrows()
):

    X, _, features = build_matrix(
        row["FeatureSet"],
        row["Scaler"],
    )

    model = KMeans(
        n_clusters=int(
            row["K"]
        ),
        n_init=50,
        random_state=
            RANDOM_STATE,
    )

    labels = model.fit_predict(
        X
    )

    exact_silhouette = (
        silhouette_score(
            X,
            labels,
        )
    )

    exact_row = {
        "FeatureSet":
            row["FeatureSet"],

        "Scaler":
            row["Scaler"],

        "K":
            int(row["K"]),

        "SilhouetteExact":
            exact_silhouette,

        "CalinskiHarabasz":
            calinski_harabasz_score(
                X,
                labels,
            ),

        "DaviesBouldin":
            davies_bouldin_score(
                X,
                labels,
            ),

        "MinClusterPct":
            (
                np.bincount(
                    labels
                ).min()
                / len(labels)
                * 100
            ),

        "MaxClusterPct":
            (
                np.bincount(
                    labels
                ).max()
                / len(labels)
                * 100
            ),
    }

    exact_rows.append(
        exact_row
    )

    print(
        f"{row['FeatureSet']} | "
        f"{row['Scaler']} | "
        f"K={int(row['K'])} | "
        f"Silhouette exacto="
        f"{exact_silhouette:.4f}"
    )


exact_df = (
    pd.DataFrame(
        exact_rows
    )
    .sort_values(
        [
            "SilhouetteExact",
            "DaviesBouldin",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)

exact_df.to_csv(
    OUTPUT_DIR /
    "kmeans_top_exact.csv",
    index=False,
)


# ============================================================
# 8. CAMPEON K-MEANS
# ============================================================

winner = exact_df.iloc[0]

winner_feature_set = (
    winner[
        "FeatureSet"
    ]
)

winner_scaler_name = (
    winner[
        "Scaler"
    ]
)

winner_k = int(
    winner[
        "K"
    ]
)

X_best, best_scaler, best_features = (
    build_matrix(
        winner_feature_set,
        winner_scaler_name,
    )
)

best_kmeans = KMeans(
    n_clusters=winner_k,
    n_init=100,
    random_state=
        RANDOM_STATE,
)

kmeans_labels = (
    best_kmeans.fit_predict(
        X_best
    )
)

kmeans_silhouette = (
    silhouette_score(
        X_best,
        kmeans_labels,
    )
)

kmeans_calinski = (
    calinski_harabasz_score(
        X_best,
        kmeans_labels,
    )
)

kmeans_davies = (
    davies_bouldin_score(
        X_best,
        kmeans_labels,
    )
)


print("\n" + "=" * 82)
print("CAMPEON K-MEANS V2")
print("=" * 82)

print(
    f"Feature set: "
    f"{winner_feature_set}"
)

print(
    f"Scaler: "
    f"{winner_scaler_name}"
)

print(
    f"K: "
    f"{winner_k}"
)

print(
    f"Silhouette exacto: "
    f"{kmeans_silhouette:.4f}"
)

print(
    f"Calinski-Harabasz: "
    f"{kmeans_calinski:.2f}"
)

print(
    f"Davies-Bouldin: "
    f"{kmeans_davies:.4f}"
)


# ============================================================
# 9. ESTABILIDAD K-MEANS
# ============================================================
#
# Comparamos la particion ganadora contra ejecuciones
# independientes mediante Adjusted Rand Index.
# ARI cercano a 1 = segmentacion muy estable.
# ============================================================

print("\n" + "=" * 82)
print("ESTABILIDAD K-MEANS")
print("=" * 82)

stability_rows = []

for seed in [
    7,
    21,
    42,
    99,
    123,
]:

    model = KMeans(
        n_clusters=winner_k,
        n_init=30,
        random_state=seed,
    )

    labels = model.fit_predict(
        X_best
    )

    ari = adjusted_rand_score(
        kmeans_labels,
        labels,
    )

    stability_rows.append(
        {
            "RandomState":
                seed,

            "AdjustedRandIndex":
                ari,
        }
    )

    print(
        f"Seed {seed:3d}: "
        f"ARI={ari:.4f}"
    )


stability_df = pd.DataFrame(
    stability_rows
)

print(
    f"\nARI promedio: "
    f"{stability_df['AdjustedRandIndex'].mean():.4f}"
)

stability_df.to_csv(
    OUTPUT_DIR /
    "kmeans_stability.csv",
    index=False,
)


# ============================================================
# 10. PERFIL K-MEANS
# ============================================================

customers[
    "KMeansClusterV2"
] = kmeans_labels

profile_columns = {
    "CustomerID":
        "count",

    "Recency":
        "mean",

    "Frequency":
        "mean",

    "Monetary":
        "mean",

    "Quantity":
        "mean",

    "UniqueProducts":
        "mean",

    "AvgOrderValue":
        "mean",
}

kmeans_profile = (
    customers
    .groupby(
        "KMeansClusterV2"
    )
    .agg(
        profile_columns
    )
    .reset_index()
)

kmeans_profile.columns = [
    "Cluster",
    "Customers",
    "Recency",
    "Frequency",
    "Monetary",
    "Quantity",
    "UniqueProducts",
    "AvgOrderValue",
]

kmeans_profile[
    "Percentage"
] = (
    kmeans_profile[
        "Customers"
    ]
    / len(customers)
    * 100
)

kmeans_profile.to_csv(
    OUTPUT_DIR /
    "kmeans_v2_cluster_profiles.csv",
    index=False,
)

print("\nPerfiles K-Means V2:")

print(
    kmeans_profile
    .round(2)
    .to_string(
        index=False
    )
)


# ============================================================
# 11. MEAN SHIFT - BANDWIDTH SEARCH
# ============================================================
#
# Se usa exactamente el MISMO espacio de features ganador
# para comparar el algoritmo, no otra representacion.
# ============================================================

print("\n" + "=" * 82)
print("MEAN SHIFT - BUSQUEDA DE BANDWIDTH")
print("=" * 82)

sample_size = min(
    3000,
    len(X_best),
)

meanshift_rows = []
meanshift_models = {}

for quantile in [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
]:

    bandwidth = estimate_bandwidth(
        X_best,
        quantile=quantile,
        n_samples=sample_size,
        random_state=
            RANDOM_STATE,
    )

    if (
        not np.isfinite(
            bandwidth
        )
        or bandwidth <= 0
    ):
        continue

    print(
        f"\nquantile={quantile:.2f} | "
        f"bandwidth={bandwidth:.4f}"
    )

    model = MeanShift(
        bandwidth=bandwidth,
        bin_seeding=True,
        cluster_all=True,
    )

    labels = model.fit_predict(
        X_best
    )

    n_clusters = len(
        np.unique(
            labels
        )
    )

    counts = np.bincount(
        labels
    )

    min_cluster_pct = (
        counts.min()
        / len(labels)
        * 100
    )

    max_cluster_pct = (
        counts.max()
        / len(labels)
        * 100
    )

    if (
        n_clusters > 1
        and n_clusters <
        len(labels)
    ):

        silhouette = (
            silhouette_score(
                X_best,
                labels,
                sample_size=min(
                    SIL_SAMPLE,
                    len(X_best),
                ),
                random_state=
                    RANDOM_STATE,
            )
        )

        calinski = (
            calinski_harabasz_score(
                X_best,
                labels,
            )
        )

        davies = (
            davies_bouldin_score(
                X_best,
                labels,
            )
        )

    else:

        silhouette = np.nan
        calinski = np.nan
        davies = np.nan

    acceptable = (
        n_clusters >= 2
        and n_clusters <= 12
        and min_cluster_pct >= 1.0
    )

    row = {
        "Quantile":
            quantile,

        "Bandwidth":
            bandwidth,

        "Clusters":
            n_clusters,

        "SilhouetteSample":
            silhouette,

        "CalinskiHarabasz":
            calinski,

        "DaviesBouldin":
            davies,

        "MinClusterPct":
            min_cluster_pct,

        "MaxClusterPct":
            max_cluster_pct,

        "AcceptableStructure":
            acceptable,
    }

    meanshift_rows.append(
        row
    )

    meanshift_models[
        quantile
    ] = (
        model,
        labels,
    )

    print(
        f"Clusters={n_clusters} | "
        f"Sil={silhouette:.4f} | "
        f"min={min_cluster_pct:.2f}% | "
        f"max={max_cluster_pct:.2f}% | "
        f"aceptable={acceptable}"
    )


meanshift_search = pd.DataFrame(
    meanshift_rows
)

meanshift_search.to_csv(
    OUTPUT_DIR /
    "meanshift_bandwidth_search.csv",
    index=False,
)


# ============================================================
# 12. ELEGIR MEAN SHIFT
# ============================================================

acceptable_ms = (
    meanshift_search[
        meanshift_search[
            "AcceptableStructure"
        ]
    ]
    .copy()
)

if len(
    acceptable_ms
) == 0:

    print(
        "\nNingun Mean Shift cumple "
        "la regla de estructura."
    )

    valid_ms = (
        meanshift_search[
            meanshift_search[
                "Clusters"
            ].between(
                2,
                20,
            )
        ]
        .dropna(
            subset=[
                "SilhouetteSample"
            ]
        )
        .copy()
    )

    selected_ms_row = (
        valid_ms
        .sort_values(
            "SilhouetteSample",
            ascending=False,
        )
        .iloc[0]
    )

else:

    selected_ms_row = (
        acceptable_ms
        .sort_values(
            [
                "SilhouetteSample",
                "DaviesBouldin",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .iloc[0]
    )


selected_quantile = float(
    selected_ms_row[
        "Quantile"
    ]
)

best_meanshift, meanshift_labels = (
    meanshift_models[
        selected_quantile
    ]
)

meanshift_silhouette = (
    silhouette_score(
        X_best,
        meanshift_labels,
    )
)

meanshift_calinski = (
    calinski_harabasz_score(
        X_best,
        meanshift_labels,
    )
)

meanshift_davies = (
    davies_bouldin_score(
        X_best,
        meanshift_labels,
    )
)

n_meanshift_clusters = len(
    np.unique(
        meanshift_labels
    )
)


print("\n" + "=" * 82)
print("MEAN SHIFT V2 SELECCIONADO")
print("=" * 82)

print(
    f"Quantile: "
    f"{selected_quantile:.2f}"
)

print(
    f"Bandwidth: "
    f"{selected_ms_row['Bandwidth']:.4f}"
)

print(
    f"Clusters: "
    f"{n_meanshift_clusters}"
)

print(
    f"Silhouette exacto: "
    f"{meanshift_silhouette:.4f}"
)

print(
    f"Calinski-Harabasz: "
    f"{meanshift_calinski:.2f}"
)

print(
    f"Davies-Bouldin: "
    f"{meanshift_davies:.4f}"
)


# ============================================================
# 13. PERFIL MEAN SHIFT
# ============================================================

customers[
    "MeanShiftClusterV2"
] = meanshift_labels

meanshift_profile = (
    customers
    .groupby(
        "MeanShiftClusterV2"
    )
    .agg(
        profile_columns
    )
    .reset_index()
)

meanshift_profile.columns = [
    "Cluster",
    "Customers",
    "Recency",
    "Frequency",
    "Monetary",
    "Quantity",
    "UniqueProducts",
    "AvgOrderValue",
]

meanshift_profile[
    "Percentage"
] = (
    meanshift_profile[
        "Customers"
    ]
    / len(customers)
    * 100
)

meanshift_profile.to_csv(
    OUTPUT_DIR /
    "meanshift_v2_cluster_profiles.csv",
    index=False,
)


# ============================================================
# 14. COMPARACION V1 / V2
# ============================================================

comparison_rows = []

if V1_FILE.exists():

    v1 = pd.read_csv(
        V1_FILE
    )

    for _, row in v1.iterrows():

        comparison_rows.append(
            {
                "Version":
                    (
                        "V1 " +
                        row["Method"]
                    ),

                "Clusters":
                    int(
                        row[
                            "Clusters"
                        ]
                    ),

                "Silhouette":
                    row[
                        "Silhouette"
                    ],

                "CalinskiHarabasz":
                    row[
                        "CalinskiHarabasz"
                    ],

                "DaviesBouldin":
                    row[
                        "DaviesBouldin"
                    ],
            }
        )


comparison_rows.extend(
    [
        {
            "Version":
                "V2 K-Means",

            "Clusters":
                winner_k,

            "Silhouette":
                kmeans_silhouette,

            "CalinskiHarabasz":
                kmeans_calinski,

            "DaviesBouldin":
                kmeans_davies,
        },

        {
            "Version":
                "V2 Mean Shift",

            "Clusters":
                n_meanshift_clusters,

            "Silhouette":
                meanshift_silhouette,

            "CalinskiHarabasz":
                meanshift_calinski,

            "DaviesBouldin":
                meanshift_davies,
        },
    ]
)

comparison = (
    pd.DataFrame(
        comparison_rows
    )
    .sort_values(
        "Silhouette",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

comparison.to_csv(
    OUTPUT_DIR /
    "v1_vs_v2_clustering.csv",
    index=False,
)


print("\n" + "=" * 82)
print("COMPARACION FINAL V1 VS V2")
print("=" * 82)

print(
    comparison
    .round(
        {
            "Silhouette": 4,
            "CalinskiHarabasz": 2,
            "DaviesBouldin": 4,
        }
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 15. MEJORA CONTRA K-MEANS V1
# ============================================================

if V1_FILE.exists():

    v1_kmeans = (
        pd.read_csv(
            V1_FILE
        )
        .query(
            "Method == 'K-Means'"
        )
        .iloc[0]
    )

    old_silhouette = float(
        v1_kmeans[
            "Silhouette"
        ]
    )

    improvement = (
        (
            kmeans_silhouette
            - old_silhouette
        )
        / old_silhouette
        * 100
    )

    print(
        f"\nCambio relativo de "
        f"Silhouette K-Means "
        f"V2 vs V1: "
        f"{improvement:+.2f}%"
    )


# ============================================================
# 16. PCA PARA VISUALIZACION
# ============================================================

pca = PCA(
    n_components=2,
    random_state=
        RANDOM_STATE,
)

X_pca = pca.fit_transform(
    X_best
)

customers[
    "PC1"
] = X_pca[:, 0]

customers[
    "PC2"
] = X_pca[:, 1]

print(
    f"\nPCA varianza explicada: "
    f"{pca.explained_variance_ratio_.sum()*100:.2f}%"
)


# K-Means
plt.figure(
    figsize=(10, 7)
)

scatter = plt.scatter(
    customers["PC1"],
    customers["PC2"],
    c=kmeans_labels,
    s=16,
    alpha=0.60,
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title(
    f"K-Means V2 "
    f"({winner_feature_set}, "
    f"K={winner_k})"
)

plt.colorbar(
    scatter,
    label="Cluster",
)

plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "01_kmeans_v2_pca.png",
    dpi=160,
)

plt.close()


# Mean Shift
plt.figure(
    figsize=(10, 7)
)

scatter = plt.scatter(
    customers["PC1"],
    customers["PC2"],
    c=meanshift_labels,
    s=16,
    alpha=0.60,
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title(
    "Mean Shift V2"
)

plt.colorbar(
    scatter,
    label="Cluster",
)

plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "02_meanshift_v2_pca.png",
    dpi=160,
)

plt.close()


# ============================================================
# 17. GUARDAR
# ============================================================

customers.to_csv(
    OUTPUT_DIR /
    "customer_segments_v2.csv",
    index=False,
)

joblib.dump(
    best_scaler,
    MODEL_DIR /
    "clustering_v2_scaler.joblib",
)

joblib.dump(
    best_kmeans,
    MODEL_DIR /
    "clustering_v2_kmeans.joblib",
)

joblib.dump(
    best_meanshift,
    MODEL_DIR /
    "clustering_v2_meanshift.joblib",
)


metadata = pd.DataFrame(
    [
        {
            "FeatureSet":
                winner_feature_set,

            "Features":
                ", ".join(
                    best_features
                ),

            "Scaler":
                winner_scaler_name,

            "K":
                winner_k,

            "KMeansSilhouette":
                kmeans_silhouette,

            "KMeansCalinski":
                kmeans_calinski,

            "KMeansDavies":
                kmeans_davies,

            "KMeansMeanARI":
                stability_df[
                    "AdjustedRandIndex"
                ].mean(),

            "MeanShiftQuantile":
                selected_quantile,

            "MeanShiftBandwidth":
                selected_ms_row[
                    "Bandwidth"
                ],

            "MeanShiftClusters":
                n_meanshift_clusters,

            "MeanShiftSilhouette":
                meanshift_silhouette,
        }
    ]
)

metadata.to_csv(
    OUTPUT_DIR /
    "clustering_v2_metadata.csv",
    index=False,
)


print("\nArchivos generados:")

for file in sorted(
    OUTPUT_DIR.glob("*")
):

    print(
        f"  {file}"
    )


print("\nSEGMENTACION V2 COMPLETADA.")
