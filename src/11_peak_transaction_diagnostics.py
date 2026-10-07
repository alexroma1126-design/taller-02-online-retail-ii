from pathlib import Path
import pandas as pd
import numpy as np


DATA_FILE = Path(
    "outputs/online_retail_clean.csv"
)

ERROR_FILE = Path(
    "outputs/prediction_v2/top_prediction_errors.csv"
)

OUTPUT_DIR = Path(
    "outputs/prediction_v2"
)

print("=" * 82)
print("DIAGNOSTICO DE TRANSACCIONES EN DIAS DE MAYOR ERROR")
print("=" * 82)


# ============================================================
# 1. CARGAR DATOS
# ============================================================

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["InvoiceDate"],
)

df["Date"] = (
    df["InvoiceDate"]
    .dt.normalize()
)

errors = pd.read_csv(
    ERROR_FILE,
    parse_dates=["Date"],
)

top_dates = (
    errors["Date"]
    .head(20)
    .tolist()
)

subset = df[
    df["Date"].isin(top_dates)
].copy()


# ============================================================
# 2. RESUMEN POR DIA
# ============================================================

daily_rows = []

for date in top_dates:

    day = subset[
        subset["Date"] == date
    ].copy()

    if day.empty:
        continue

    total_sales = (
        day["Sales"].sum()
    )

    # --------------------------------------------------------
    # Mayor linea individual
    # --------------------------------------------------------

    max_line_idx = (
        day["Sales"].idxmax()
    )

    max_line = day.loc[
        max_line_idx
    ]

    max_line_sales = float(
        max_line["Sales"]
    )

    max_line_share = (
        max_line_sales /
        total_sales *
        100
        if total_sales != 0
        else np.nan
    )


    # --------------------------------------------------------
    # Mayor factura
    # --------------------------------------------------------

    invoices = (
        day.groupby("Invoice")["Sales"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    max_invoice_sales = float(
        invoices.iloc[0]
    )

    max_invoice_share = (
        max_invoice_sales /
        total_sales *
        100
        if total_sales != 0
        else np.nan
    )


    # --------------------------------------------------------
    # Mayor cliente
    # --------------------------------------------------------

    customers = (
        day.groupby("CustomerID")["Sales"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    max_customer_sales = float(
        customers.iloc[0]
    )

    max_customer_share = (
        max_customer_sales /
        total_sales *
        100
        if total_sales != 0
        else np.nan
    )


    daily_rows.append(
        {
            "Date":
                date,

            "DayOfWeek":
                date.day_name(),

            "DailySales":
                total_sales,

            "Lines":
                len(day),

            "Invoices":
                day["Invoice"].nunique(),

            "Customers":
                day["CustomerID"].nunique(),

            "MaxLineSales":
                max_line_sales,

            "MaxLineSharePct":
                max_line_share,

            "MaxInvoiceSales":
                max_invoice_sales,

            "MaxInvoiceSharePct":
                max_invoice_share,

            "MaxCustomerSales":
                max_customer_sales,

            "MaxCustomerSharePct":
                max_customer_share,

            "TopStockCode":
                str(
                    max_line["StockCode"]
                ),

            "TopDescription":
                str(
                    max_line["Description"]
                ),

            "TopQuantity":
                max_line["Quantity"],

            "TopPrice":
                max_line["Price"],
        }
    )


summary = pd.DataFrame(
    daily_rows
)


# ============================================================
# 3. UNIR ERROR DEL MODELO
# ============================================================

error_columns = errors[
    [
        "Date",
        "ActualSales",
        "extra_trees_calendar",
        "AbsoluteError",
        "PercentageContribution",
    ]
].copy()

summary = summary.merge(
    error_columns,
    on="Date",
    how="left",
)

summary = summary.sort_values(
    "AbsoluteError",
    ascending=False,
)


# ============================================================
# 4. MOSTRAR
# ============================================================

print("\n" + "=" * 82)
print("CONCENTRACION DE VENTAS EN LOS DIAS DE MAYOR ERROR")
print("=" * 82)

columns = [
    "Date",
    "DayOfWeek",
    "DailySales",
    "MaxLineSales",
    "MaxLineSharePct",
    "MaxInvoiceSales",
    "MaxInvoiceSharePct",
    "MaxCustomerSharePct",
    "AbsoluteError",
]

print(
    summary[
        columns
    ]
    .round(2)
    .to_string(
        index=False
    )
)


# ============================================================
# 5. DIA DE MAYOR ERROR
# ============================================================

worst_date = (
    summary.iloc[0]["Date"]
)

worst_day = df[
    df["Date"] == worst_date
].copy()

worst_day = worst_day.sort_values(
    "Sales",
    ascending=False,
)


print("\n" + "=" * 82)
print(
    f"DETALLE DEL DIA DE MAYOR ERROR: "
    f"{worst_date.date()}"
)
print("=" * 82)

print(
    worst_day[
        [
            "Invoice",
            "StockCode",
            "Description",
            "Quantity",
            "Price",
            "Sales",
            "CustomerID",
        ]
    ]
    .head(15)
    .round(2)
    .to_string(
        index=False
    )
)


# ============================================================
# 6. MEDIR CUANTOS DIAS ESTAN DOMINADOS POR UN PEDIDO
# ============================================================

for threshold in [
    25,
    50,
    75,
    90,
]:

    count = (
        summary[
            "MaxInvoiceSharePct"
        ] >= threshold
    ).sum()

    print(
        f"\nDias donde una sola factura representa "
        f">= {threshold}% de las ventas: "
        f"{count}/{len(summary)}"
    )


# ============================================================
# 7. GUARDAR
# ============================================================

summary.to_csv(
    OUTPUT_DIR /
    "peak_transaction_diagnostics.csv",
    index=False,
)

worst_day.to_csv(
    OUTPUT_DIR /
    "worst_error_day_transactions.csv",
    index=False,
)


print(
    "\nDIAGNOSTICO DE PICOS COMPLETADO."
)
