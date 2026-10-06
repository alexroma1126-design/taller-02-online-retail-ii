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

## Validacion

El proyecto incluye una auditoria automatica.

Resultado final:

**AUDITORIA SUPERADA: 26 verificaciones OK**

El Taller 02 esta computacionalmente consistente.

## Estructura

- `src/01_prepare_data.py`: preparacion y limpieza.
- `src/02_eda.py`: analisis exploratorio.
- `src/03_customer_classification.py`: clasificacion.
- `src/04_customer_segmentation.py`: clustering.
- `src/05_sales_prediction.py`: prediccion.
- `src/06_validate_project.py`: auditoria final.
- `figures/`: figuras generadas.
- `outputs/`: tablas y resultados.
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
