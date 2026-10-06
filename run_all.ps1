$ErrorActionPreference = "Stop"

Write-Host "=== TALLER 02 - EJECUCION COMPLETA ==="

python src\01_prepare_data.py
python src\02_eda.py
python src\03_customer_classification.py
python src\04_customer_segmentation.py
python src\05_sales_prediction.py
python src\06_validate_project.py

Write-Host ""
Write-Host "=== EJECUCION FINALIZADA CORRECTAMENTE ==="
