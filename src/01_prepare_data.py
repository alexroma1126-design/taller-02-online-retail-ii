from pathlib import Path
import pandas as pd


# ============================================================
# TALLER 02 - MAESTRIA EN CIENCIA DE DATOS
# ETAPA 01: CARGA, EXPLORACION Y LIMPIEZA
# Dataset: Online Retail II
# ============================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("TALLER 02 - PREPARACION DEL DATASET ONLINE RETAIL II")
print("=" * 70)


# ------------------------------------------------------------
# 1. LOCALIZAR EL ARCHIVO EXCEL
# ------------------------------------------------------------

excel_files = list(DATA_DIR.glob("*.xlsx")) + list(DATA_DIR.glob("*.xls"))

if not excel_files:
    raise FileNotFoundError(
        "No se encontro ningun archivo Excel dentro de la carpeta data/"
    )

excel_path = excel_files[0]

print(f"\nArchivo encontrado:")
print(f"  {excel_path}")


# ------------------------------------------------------------
# 2. LEER TODAS LAS HOJAS
# ------------------------------------------------------------

xls = pd.ExcelFile(excel_path)

print("\nHojas encontradas:")
for sheet in xls.sheet_names:
    print(f"  - {sheet}")

frames = []

for sheet in xls.sheet_names:
    print(f"\nLeyendo hoja: {sheet}")

    temp = pd.read_excel(
        excel_path,
        sheet_name=sheet
    )

    temp["SourceSheet"] = sheet

    print(f"  Filas:    {len(temp):,}")
    print(f"  Columnas: {len(temp.columns)}")

    frames.append(temp)

df = pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------
# 3. NORMALIZAR NOMBRES DE COLUMNAS
# ------------------------------------------------------------

rename_map = {
    "Invoice": "Invoice",
    "InvoiceNo": "Invoice",
    "StockCode": "StockCode",
    "Description": "Description",
    "Quantity": "Quantity",
    "InvoiceDate": "InvoiceDate",
    "Price": "Price",
    "UnitPrice": "Price",
    "Customer ID": "CustomerID",
    "CustomerID": "CustomerID",
    "Country": "Country",
}

df = df.rename(
    columns={
        col: rename_map[col]
        for col in df.columns
        if col in rename_map
    }
)


required_columns = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "CustomerID",
    "Country",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Faltan columnas esperadas: {missing_columns}\n"
        f"Columnas disponibles: {df.columns.tolist()}"
    )


# ------------------------------------------------------------
# 4. EXPLORACION INICIAL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ESTRUCTURA INICIAL")
print("=" * 70)

print(f"\nFilas totales:    {len(df):,}")
print(f"Columnas totales: {len(df.columns)}")

print("\nColumnas:")
for col in df.columns:
    print(f"  - {col}")

print("\nPrimeras 5 observaciones:")
print(df.head().to_string(index=False))

print("\nTipos de datos:")
print(df.dtypes)

print("\nValores faltantes:")
missing = df.isna().sum().sort_values(ascending=False)
print(missing.to_string())


# ------------------------------------------------------------
# 5. LIMPIEZA
# ------------------------------------------------------------

initial_rows = len(df)

# CustomerID faltante
missing_customer = df["CustomerID"].isna().sum()

# Facturas canceladas:
# en Online Retail II normalmente empiezan con "C"
invoice_str = df["Invoice"].astype(str)
cancelled_mask = invoice_str.str.startswith("C", na=False)
cancelled_rows = cancelled_mask.sum()

# Duplicados exactos
duplicate_rows = df.duplicated().sum()

# Cantidad no positiva
nonpositive_quantity = (df["Quantity"] <= 0).sum()

# Precio no positivo
nonpositive_price = (df["Price"] <= 0).sum()


print("\n" + "=" * 70)
print("PROBLEMAS DETECTADOS")
print("=" * 70)

print(f"CustomerID faltante:        {missing_customer:,}")
print(f"Transacciones canceladas:   {cancelled_rows:,}")
print(f"Duplicados exactos:         {duplicate_rows:,}")
print(f"Quantity <= 0:              {nonpositive_quantity:,}")
print(f"Price <= 0:                 {nonpositive_price:,}")


# ------------------------------------------------------------
# Aplicar filtros
# ------------------------------------------------------------

clean = df.copy()

# Eliminar duplicados
clean = clean.drop_duplicates()

# Eliminar clientes sin identificacion
clean = clean.dropna(subset=["CustomerID"])

# Eliminar facturas canceladas
clean = clean[
    ~clean["Invoice"]
    .astype(str)
    .str.startswith("C", na=False)
]

# Eliminar cantidades no positivas
clean = clean[
    clean["Quantity"] > 0
]

# Eliminar precios no positivos
clean = clean[
    clean["Price"] > 0
]

# Convertir fecha
clean["InvoiceDate"] = pd.to_datetime(
    clean["InvoiceDate"],
    errors="coerce"
)

clean = clean.dropna(subset=["InvoiceDate"])

# CustomerID como entero
clean["CustomerID"] = clean["CustomerID"].astype(int)

# Variable monetaria fundamental
clean["Sales"] = (
    clean["Quantity"] *
    clean["Price"]
)


# ------------------------------------------------------------
# 6. CARACTERISTICAS TEMPORALES BASICAS
# ------------------------------------------------------------

clean["Year"] = clean["InvoiceDate"].dt.year
clean["Month"] = clean["InvoiceDate"].dt.month
clean["Day"] = clean["InvoiceDate"].dt.day
clean["DayOfWeek"] = clean["InvoiceDate"].dt.dayofweek
clean["Hour"] = clean["InvoiceDate"].dt.hour

clean["YearMonth"] = (
    clean["InvoiceDate"]
    .dt.to_period("M")
    .astype(str)
)


# ------------------------------------------------------------
# 7. RESUMEN DEL DATASET LIMPIO
# ------------------------------------------------------------

final_rows = len(clean)

print("\n" + "=" * 70)
print("DATASET LIMPIO")
print("=" * 70)

print(f"\nFilas iniciales:  {initial_rows:,}")
print(f"Filas finales:    {final_rows:,}")
print(f"Filas eliminadas: {initial_rows - final_rows:,}")

print(f"\nClientes unicos:  {clean['CustomerID'].nunique():,}")
print(f"Facturas unicas:  {clean['Invoice'].nunique():,}")
print(f"Productos unicos: {clean['StockCode'].nunique():,}")
print(f"Paises:           {clean['Country'].nunique():,}")

print("\nPeriodo:")
print(f"  Desde: {clean['InvoiceDate'].min()}")
print(f"  Hasta: {clean['InvoiceDate'].max()}")

print("\nVentas totales:")
print(f"  £{clean['Sales'].sum():,.2f}")

print("\nEstadisticas numericas:")
print(
    clean[
        ["Quantity", "Price", "Sales"]
    ]
    .describe()
    .round(2)
    .to_string()
)


# ------------------------------------------------------------
# 8. GUARDAR DATASET LIMPIO
# ------------------------------------------------------------

output_path = OUTPUT_DIR / "online_retail_clean.csv"

clean.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 70)
print("ARCHIVO GENERADO")
print("=" * 70)

print(f"\n{output_path}")
print(f"Tamaño: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

print("\nETAPA 01 COMPLETADA CORRECTAMENTE.")
