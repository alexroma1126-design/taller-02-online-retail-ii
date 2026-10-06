from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# TALLER 02 - MAESTRIA EN CIENCIA DE DATOS
# ETAPA 02: ANALISIS EXPLORATORIO DE DATOS
# ============================================================

INPUT_FILE = Path("outputs/online_retail_clean.csv")
FIG_DIR = Path("figures/eda")
OUT_DIR = Path("outputs/eda")

FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("TALLER 02 - ANALISIS EXPLORATORIO DE DATOS")
print("=" * 72)


# ------------------------------------------------------------
# 1. CARGAR DATOS LIMPIOS
# ------------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["InvoiceDate"]
)

print(f"\nFilas cargadas: {len(df):,}")
print(f"Clientes:       {df['CustomerID'].nunique():,}")
print(f"Productos:      {df['StockCode'].nunique():,}")
print(f"Ventas totales: £{df['Sales'].sum():,.2f}")


# ------------------------------------------------------------
# 2. RESUMEN ESTADISTICO
# ------------------------------------------------------------

summary = df[
    ["Quantity", "Price", "Sales"]
].describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
)

summary.to_csv(
    OUT_DIR / "01_numeric_summary.csv"
)

print("\nResumen estadistico:")
print(summary.round(2).to_string())


# ------------------------------------------------------------
# 3. DISTRIBUCIONES
# Para evitar que valores extremos destruyan visualmente
# los histogramas usamos el percentil 99.
# ------------------------------------------------------------

quantity_limit = df["Quantity"].quantile(0.99)
price_limit = df["Price"].quantile(0.99)
sales_limit = df["Sales"].quantile(0.99)

print("\nPercentiles 99:")
print(f"Quantity: {quantity_limit:,.2f}")
print(f"Price:    £{price_limit:,.2f}")
print(f"Sales:    £{sales_limit:,.2f}")


def save_histogram(series, xlabel, title, filename, upper_limit):
    filtered = series[
        (series >= 0) &
        (series <= upper_limit)
    ]

    plt.figure(figsize=(10, 6))
    plt.hist(filtered, bins=50, edgecolor="black")
    plt.xlabel(xlabel)
    plt.ylabel("Frecuencia")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=160)
    plt.close()


save_histogram(
    df["Quantity"],
    "Cantidad comprada",
    "Distribucion de cantidades compradas (hasta P99)",
    "01_quantity_distribution.png",
    quantity_limit,
)

save_histogram(
    df["Price"],
    "Precio unitario (£)",
    "Distribucion de precios unitarios (hasta P99)",
    "02_price_distribution.png",
    price_limit,
)

save_histogram(
    df["Sales"],
    "Venta por linea (£)",
    "Distribucion del valor de las transacciones (hasta P99)",
    "03_sales_distribution.png",
    sales_limit,
)


# ------------------------------------------------------------
# 4. PRODUCTOS MAS VENDIDOS POR UNIDADES
# ------------------------------------------------------------

product_quantity = (
    df.groupby(
        ["StockCode", "Description"],
        dropna=False
    )["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

product_quantity.to_csv(
    OUT_DIR / "02_top_products_by_quantity.csv",
    index=False
)

top_quantity = product_quantity.head(15).copy()

print("\n" + "=" * 72)
print("TOP 15 PRODUCTOS POR UNIDADES")
print("=" * 72)

print(
    top_quantity[
        ["StockCode", "Description", "Quantity"]
    ].to_string(index=False)
)

plot_q = top_quantity.iloc[::-1]

plt.figure(figsize=(11, 8))
plt.barh(
    plot_q["Description"].fillna(plot_q["StockCode"].astype(str)),
    plot_q["Quantity"]
)
plt.xlabel("Unidades vendidas")
plt.ylabel("Producto")
plt.title("15 productos mas vendidos por cantidad")
plt.tight_layout()
plt.savefig(
    FIG_DIR / "04_top_products_quantity.png",
    dpi=160
)
plt.close()


# ------------------------------------------------------------
# 5. PRODUCTOS CON MAYORES INGRESOS
# ------------------------------------------------------------

product_sales = (
    df.groupby(
        ["StockCode", "Description"],
        dropna=False
    )["Sales"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

product_sales.to_csv(
    OUT_DIR / "03_top_products_by_sales.csv",
    index=False
)

top_sales = product_sales.head(15).copy()

print("\n" + "=" * 72)
print("TOP 15 PRODUCTOS POR INGRESOS")
print("=" * 72)

print(
    top_sales[
        ["StockCode", "Description", "Sales"]
    ]
    .round({"Sales": 2})
    .to_string(index=False)
)

plot_s = top_sales.iloc[::-1]

plt.figure(figsize=(11, 8))
plt.barh(
    plot_s["Description"].fillna(plot_s["StockCode"].astype(str)),
    plot_s["Sales"]
)
plt.xlabel("Ventas (£)")
plt.ylabel("Producto")
plt.title("15 productos con mayores ingresos")
plt.tight_layout()
plt.savefig(
    FIG_DIR / "05_top_products_sales.png",
    dpi=160
)
plt.close()


# ------------------------------------------------------------
# 6. VENTAS MENSUALES
# ------------------------------------------------------------

df["YearMonthDate"] = (
    df["InvoiceDate"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

monthly = (
    df.groupby("YearMonthDate")
    .agg(
        Sales=("Sales", "sum"),
        Quantity=("Quantity", "sum"),
        Invoices=("Invoice", "nunique"),
        Customers=("CustomerID", "nunique")
    )
    .reset_index()
    .sort_values("YearMonthDate")
)

monthly.to_csv(
    OUT_DIR / "04_monthly_sales.csv",
    index=False
)

print("\n" + "=" * 72)
print("VENTAS MENSUALES")
print("=" * 72)

print(
    monthly
    .round({"Sales": 2})
    .to_string(index=False)
)

plt.figure(figsize=(12, 6))
plt.plot(
    monthly["YearMonthDate"],
    monthly["Sales"],
    marker="o"
)
plt.xlabel("Mes")
plt.ylabel("Ventas (£)")
plt.title("Evolucion mensual de las ventas")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    FIG_DIR / "06_monthly_sales_trend.png",
    dpi=160
)
plt.close()


# ------------------------------------------------------------
# 7. ESTACIONALIDAD POR MES DEL AÑO
# ------------------------------------------------------------

df["MonthNumber"] = df["InvoiceDate"].dt.month

month_names = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre",
}

season_month = (
    df.groupby("MonthNumber")
    .agg(
        Sales=("Sales", "sum"),
        Quantity=("Quantity", "sum"),
        Invoices=("Invoice", "nunique")
    )
    .reset_index()
)

season_month["Month"] = (
    season_month["MonthNumber"]
    .map(month_names)
)

season_month.to_csv(
    OUT_DIR / "05_sales_by_month_of_year.csv",
    index=False
)

plt.figure(figsize=(11, 6))
plt.bar(
    season_month["Month"],
    season_month["Sales"]
)
plt.xlabel("Mes")
plt.ylabel("Ventas acumuladas (£)")
plt.title("Estacionalidad de ventas por mes del año")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    FIG_DIR / "07_sales_by_month.png",
    dpi=160
)
plt.close()


# ------------------------------------------------------------
# 8. ESTACIONALIDAD POR DIA DE LA SEMANA
# ------------------------------------------------------------

df["WeekdayNumber"] = df["InvoiceDate"].dt.dayofweek

weekday_names = {
    0: "Lunes",
    1: "Martes",
    2: "Miercoles",
    3: "Jueves",
    4: "Viernes",
    5: "Sabado",
    6: "Domingo",
}

weekday = (
    df.groupby("WeekdayNumber")
    .agg(
        Sales=("Sales", "sum"),
        Quantity=("Quantity", "sum"),
        Invoices=("Invoice", "nunique")
    )
    .reset_index()
)

weekday["Weekday"] = (
    weekday["WeekdayNumber"]
    .map(weekday_names)
)

weekday.to_csv(
    OUT_DIR / "06_sales_by_weekday.csv",
    index=False
)

plt.figure(figsize=(10, 6))
plt.bar(
    weekday["Weekday"],
    weekday["Sales"]
)
plt.xlabel("Dia")
plt.ylabel("Ventas (£)")
plt.title("Ventas segun dia de la semana")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(
    FIG_DIR / "08_sales_by_weekday.png",
    dpi=160
)
plt.close()


# ------------------------------------------------------------
# 9. PAISES
# ------------------------------------------------------------

country = (
    df.groupby("Country")
    .agg(
        Sales=("Sales", "sum"),
        Quantity=("Quantity", "sum"),
        Customers=("CustomerID", "nunique")
    )
    .sort_values("Sales", ascending=False)
    .reset_index()
)

country.to_csv(
    OUT_DIR / "07_sales_by_country.csv",
    index=False
)

print("\n" + "=" * 72)
print("TOP 10 PAISES POR VENTAS")
print("=" * 72)

print(
    country.head(10)
    .round({"Sales": 2})
    .to_string(index=False)
)


# ------------------------------------------------------------
# 10. PERIODOS DE MAXIMAS VENTAS
# ------------------------------------------------------------

best_month = monthly.loc[
    monthly["Sales"].idxmax()
]

best_day = (
    df.assign(
        Date=df["InvoiceDate"].dt.date
    )
    .groupby("Date")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

best_hour = (
    df.groupby(
        df["InvoiceDate"].dt.hour
    )["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 72)
print("PERIODOS DE MAXIMAS VENTAS")
print("=" * 72)

print(
    f"Mes con mayor venta: "
    f"{best_month['YearMonthDate']:%Y-%m} "
    f"-> £{best_month['Sales']:,.2f}"
)

print(
    f"Dia con mayor venta: "
    f"{best_day.index[0]} "
    f"-> £{best_day.iloc[0]:,.2f}"
)

print(
    f"Hora con mayor venta acumulada: "
    f"{int(best_hour.index[0]):02d}:00 "
    f"-> £{best_hour.iloc[0]:,.2f}"
)


# ------------------------------------------------------------
# 11. ARCHIVO RESUMEN
# ------------------------------------------------------------

report_lines = [
    "TALLER 02 - RESUMEN EDA",
    "=" * 60,
    f"Transacciones limpias: {len(df):,}",
    f"Clientes: {df['CustomerID'].nunique():,}",
    f"Facturas: {df['Invoice'].nunique():,}",
    f"Productos: {df['StockCode'].nunique():,}",
    f"Paises: {df['Country'].nunique():,}",
    f"Ventas totales: £{df['Sales'].sum():,.2f}",
    "",
    f"Mes de mayores ventas: {best_month['YearMonthDate']:%Y-%m}",
    f"Ventas del mejor mes: £{best_month['Sales']:,.2f}",
    f"Dia de mayores ventas: {best_day.index[0]}",
    f"Ventas del mejor dia: £{best_day.iloc[0]:,.2f}",
    f"Hora de mayores ventas acumuladas: {int(best_hour.index[0]):02d}:00",
]

report_path = OUT_DIR / "EDA_SUMMARY.txt"

report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print("\n" + "=" * 72)
print("ARCHIVOS GENERADOS")
print("=" * 72)

print(f"\nFiguras:")
for file in sorted(FIG_DIR.glob("*.png")):
    print(f"  {file}")

print("\nTablas:")
for file in sorted(OUT_DIR.glob("*")):
    print(f"  {file}")

print("\nETAPA 02 COMPLETADA CORRECTAMENTE.")
