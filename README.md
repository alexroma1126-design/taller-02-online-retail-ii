# Taller 02 - Aprendizaje de Maquina para el Analisis del Rendimiento de Ventas

**Maestria en Ciencia de Datos - Universidad Yachay Tech**

Proyecto desarrollado con el conjunto de datos **Online Retail II** del UCI Machine Learning Repository.

El trabajo comprende:

1. Preparacion y limpieza de datos.
2. Analisis exploratorio de datos.
3. Clasificacion de clientes Normal/Premium.
4. Segmentacion mediante K-Means y Mean Shift.
5. Prediccion de ventas.
6. Validacion automatica del proyecto.

## Dataset

Periodo original: diciembre de 2009 a diciembre de 2011.

Resultados despues de la limpieza:

| Metrica | Resultado |
|---|---:|
| Transacciones originales | 1,067,371 |
| Transacciones limpias | 793,609 |
| Clientes unicos | 5,878 |
| Facturas unicas | 36,969 |
| Productos unicos | 4,631 |
| Paises | 41 |
| Ventas totales | GBP 17,685,460.64 |

## Analisis Exploratorio

Se analizaron las distribuciones de cantidad, precio y ventas, los productos mas vendidos, la evolucion temporal y los patrones estacionales.

El mes con mayores ventas fue **noviembre de 2010**, con aproximadamente **GBP 1,166,460.02**.

Las figuras del EDA se encuentran en `figures/eda/`.

## Clasificacion de clientes

Se definieron dos clases:

- Normal
- Premium

Para evitar fuga de informacion se utilizo una division temporal.

Las caracteristicas del cliente se calcularon utilizando el periodo historico anterior al 1 de diciembre de 2010. La categoria Premium se definio usando el comportamiento posterior.

Un cliente Premium corresponde al 25 % superior segun ventas futuras.

Umbral Premium: **GBP 2,246.21**.

Distribucion:

- Normal: 2,044 clientes.
- Premium: 682 clientes.

Se compararon Logistic Regression y Random Forest.

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.8407 | 0.6623 | 0.7445 | 0.7010 | 0.8756 |
| Logistic Regression | 0.8187 | 0.6118 | 0.7591 | 0.6775 | 0.8655 |

El mejor modelo segun F1 fue **Random Forest**.

## Segmentacion de clientes

Se utilizaron:

- Recency
- Frequency
- Monetary
- Quantity
- UniqueProducts
- AvgOrderValue

K-Means fue evaluado para K entre 2 y 8.

El mejor resultado segun Silhouette fue **K = 2**.

| Metodo | Clusters | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---|---:|---:|---:|---:|
| K-Means | 2 | 0.3640 | 4549.4038 | 1.0257 |
| Mean Shift | 8 | 0.1811 | 253.2709 | 1.1055 |

K-Means genero dos grandes perfiles:

- clientes ocasionales o de baja actividad;
- clientes activos y de mayor valor.

Mean Shift encontro ocho grupos, aunque el 92.48 % de los clientes se concentro en un unico cluster.

La representacion PCA en dos componentes conserva aproximadamente **83.76 % de la varianza**.

## Prediccion de ventas

La variable objetivo corresponde a las **ventas diarias**.

Se utilizaron caracteristicas temporales, variables ciclicas, rezagos de ventas y medias moviles.

Entre ellas:

- SalesLag1
- SalesLag7
- SalesLag14
- SalesLag28
- SalesRolling7
- SalesRolling14
- SalesRolling28

La division entrenamiento/prueba se realizo cronologicamente.

Entrenamiento:

2009-12-29 a 2011-07-19.

Prueba:

2011-07-20 a 2011-12-09.

Resultados:

| Modelo | MAE | RMSE | R2 | sMAPE |
|---|---:|---:|---:|---:|
| Random Forest | 10,761.00 | 18,308.38 | 0.4170 | 44.65 |
| Linear Regression | 11,485.25 | 18,054.00 | 0.4331 | 56.12 |
| Decision Tree | 13,220.72 | 21,271.99 | 0.2130 | 41.04 |
| Baseline Lag-1 | 20,091.79 | 26,551.33 | -0.2262 | 90.08 |

El modelo seleccionado por MAE fue **Random Forest**.

Su MAE mejora aproximadamente **46.44 %** respecto al baseline.

Las variables mas importantes fueron SalesLag7 y SalesRolling7, mostrando una componente semanal importante en las ventas.


## Optimizacion y seleccion final de modelos

Ademas de los modelos iniciales, se realizo una segunda etapa experimental
para comprobar si era posible mejorar los resultados sin modificar
artificialmente los conjuntos de evaluacion.

La seleccion final se realizo manteniendo trazabilidad entre las versiones V1,
V2 y los experimentos posteriores.

### Clasificacion de clientes

Se conservaron exactamente:

- la fecha de corte temporal del modelo original;
- los 2,726 clientes presentes en ambos periodos;
- las nueve variables predictoras originales;
- la definicion Premium basada en el percentil 75 de ventas futuras;
- el mismo split estratificado de 2,180 clientes de entrenamiento y
  546 clientes de prueba.

Sobre el conjunto de entrenamiento se compararon:

- Logistic Regression;
- Random Forest;
- Extra Trees;
- HistGradientBoosting.

La seleccion del modelo y del threshold se realizo mediante validacion cruzada
estratificada y predicciones out-of-fold, sin utilizar el test final.

Logistic Regression fue el mejor candidato durante la seleccion:

| Modelo | F1 OOF | ROC-AUC OOF | PR-AUC OOF |
|---|---:|---:|---:|
| Logistic Regression | 0.6990 | 0.8876 | 0.7714 |
| HistGradientBoosting | 0.6927 | 0.8823 | 0.7615 |
| Extra Trees | 0.6864 | 0.8815 | 0.7627 |
| Random Forest ajustado | 0.6815 | 0.8743 | 0.7466 |

El threshold de Logistic Regression fue fijado en **0.52** antes de evaluar
el test final.

En el test reservado:

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest V1 | 0.8407 | 0.6623 | 0.7445 | **0.7010** | **0.8756** |
| Logistic Regression V2 | 0.8370 | 0.6538 | 0.7445 | 0.6962 | 0.8692 |

Por tanto, **Random Forest V1 se conserva como modelo final de
clasificacion**. La experimentacion V2 no produjo una mejora fuera de muestra.

### Segmentacion de clientes

La version inicial utilizaba seis variables:

- Recency;
- Frequency;
- Monetary;
- Quantity;
- UniqueProducts;
- AvgOrderValue.

El analisis de correlacion mostro una fuerte redundancia entre algunas
variables. Por ejemplo, la correlacion de Spearman entre Monetary y Quantity
fue aproximadamente **0.941**.

Se compararon cuatro representaciones:

- RFM;
- RFM + UniqueProducts;
- RFM + Quantity + UniqueProducts;
- las seis variables originales.

Tambien se compararon StandardScaler y RobustScaler para K entre 2 y 8.

La mejor representacion fue el **RFM clasico**:

- Recency;
- Frequency;
- Monetary.

con **StandardScaler y K = 2**.

Resultados:

| Version | Clusters | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---|---:|---:|---:|---:|
| K-Means V1 | 2 | 0.3640 | 4549.40 | 1.0257 |
| **K-Means V2 RFM** | **2** | **0.4386** | **6185.15** | **0.8727** |
| Mean Shift V1 | 8 | 0.1811 | 253.27 | 1.1055 |
| Mean Shift V2 | 2 | 0.4246 | 2328.71 | 0.7391 |

K-Means V2 mejora el Silhouette aproximadamente **20.50 %** respecto a V1.

Ademas, la estabilidad frente a diferentes semillas fue muy alta:

**Adjusted Rand Index promedio = 0.9970**.

Los dos segmentos obtenidos son:

| Cluster | Clientes | Porcentaje | Recency | Frequency | Monetary |
|---|---:|---:|---:|---:|---:|
| 0 | 2,316 | 39.4 % | 49.96 | 12.73 | GBP 6,671.32 |
| 1 | 3,562 | 60.6 % | 299.75 | 2.10 | GBP 627.37 |

El primer cluster representa clientes mas activos y de mayor valor, mientras
que el segundo agrupa clientes menos frecuentes y con una mayor recencia.

Este resultado muestra que agregar mas variables no necesariamente mejora
K-Means: una representacion RFM mas parsimoniosa produjo clusters mas separados
y estables.

### Prediccion de ventas

La evaluacion de forecasting fue ampliada incorporando baselines mas exigentes,
nuevas variables temporales y modelos adicionales.

Se agregaron los baselines:

- Lag-1;
- Lag-7;
- media movil de siete dias.

El baseline Lag-7 fue considerablemente mas fuerte que Lag-1, confirmando una
estructura semanal importante en las ventas.

Se compararon:

- Random Forest ajustado;
- Extra Trees;
- HistGradientBoosting.

La comparacion final se realizo sobre exactamente el mismo periodo de prueba de
V1:

**20/07/2011 - 09/12/2011, 143 dias.**

Resultados principales:

| Modelo | MAE | RMSE | R2 | sMAPE |
|---|---:|---:|---:|---:|
| Random Forest V1 | 10,761.00 | 18,308.38 | 0.4170 | 44.65 |
| Baseline Lag-7 | 12,063.21 | 20,192.69 | 0.2908 | 39.05 |
| **Extra Trees V2 + calendario** | **9,576.10** | **16,540.52** | **0.5241** | **31.38** |

Extra Trees V2 reduce el MAE aproximadamente **11.01 %** respecto a Random
Forest V1 y aproximadamente **20.6 %** frente al baseline Lag-7.

El analisis de dias sin ventas encontro un patron operativo especialmente
fuerte:

- 84 de 85 sabados del periodo de entrenamiento tuvieron ventas iguales a cero;
- los 20 sabados del test tambien tuvieron ventas iguales a cero.

La regla de calendario fue aprendida exclusivamente a partir del periodo de
entrenamiento y se utilizo para fijar la prediccion de los dias
estructuralmente cerrados en cero.

Esta correccion tuvo poco impacto sobre MAE, pero redujo el sMAPE de
aproximadamente **53.94 % a 31.38 %**.

Tambien se analizaron errores extremos. El 09/12/2011 registro ventas por
GBP 184,347.66, de las cuales una sola linea de factura represento
GBP 168,469.60, aproximadamente el 91.39 % del total diario.

El evento se mantuvo dentro del test: no se eliminaron observaciones dificiles
para mejorar artificialmente las metricas.

Una V3 con 65 variables, tendencias semanales y variables operativas adicionales
fue evaluada posteriormente, pero no supero a V2:

| Version | MAE | RMSE | R2 | sMAPE |
|---|---:|---:|---:|---:|
| V2 Extra Trees + calendario | **9,576.10** | **16,540.52** | **0.5241** | **31.38** |
| V3 Extra Trees Absolute + calendario | 9,691.55 | 17,109.36 | 0.4909 | 31.57 |

Por parsimonia y rendimiento fuera de muestra se conserva **V2 Extra Trees +
calendario** como modelo final de prediccion.

### Resumen de modelos finales

| Problema | Modelo inicial | Modelo final | Resultado |
|---|---|---|---|
| Clasificacion | Random Forest | **Random Forest V1** | F1 = 0.7010 |
| Segmentacion | K-Means con 6 variables | **K-Means RFM, K=2** | Silhouette = 0.4386 |
| Prediccion | Random Forest | **Extra Trees + calendario** | MAE = 9,576.10 |

Los experimentos completos y los resultados consolidados se encuentran en
`outputs/champion/`.


## Validacion

El proyecto incluye una auditoria automatica.

Resultado final:

**AUDITORIA SUPERADA: 26 verificaciones OK**

El Taller 02 esta computacionalmente consistente.

## Estructura

Pipeline original:

- `src/01_prepare_data.py`: preparacion y limpieza.
- `src/02_eda.py`: analisis exploratorio.
- `src/03_customer_classification.py`: clasificacion V1.
- `src/04_customer_segmentation.py`: clustering V1.
- `src/05_sales_prediction.py`: prediccion V1.
- `src/06_validate_project.py`: auditoria del pipeline original.

Experimentacion y mejora:

- `src/07_sales_prediction_v2.py`: forecasting V2.
- `src/08_sales_prediction_v2_fair.py`: comparacion V1/V2 sobre el mismo test.
- `src/09_zero_day_diagnostics.py`: diagnostico de dias sin ventas.
- `src/10_residual_diagnostics.py`: analisis de residuos.
- `src/11_peak_transaction_diagnostics.py`: diagnostico de ventas extremas.
- `src/12_sales_prediction_v3.py`: experimento V3.
- `src/13_customer_classification_v2_selection.py`: seleccion OOF de clasificacion.
- `src/14_customer_classification_v2_final_test.py`: evaluacion final reservada.
- `src/15_customer_segmentation_v2.py`: optimizacion de clustering.
- `src/16_final_model_summary.py`: resumen consolidado.
- `src/17_validate_improvements.py`: auditoria de resultados finales.
- `figures/`: figuras generadas.
- `outputs/champion/`: resultados seleccionados.
- `outputs/`: resultados y diagnosticos completos.
- `models/`: modelos entrenados.

## Reproducibilidad

Instalar dependencias:

    python -m pip install -r requirements.txt

Ejecutar todo el proyecto:

    .\run_all.ps1

O ejecutar cada etapa individualmente desde la terminal.

## Fuente

Dataset: **Online Retail II**

UCI Machine Learning Repository.

Periodo: 01/12/2009 - 09/12/2011.
