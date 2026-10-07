from pathlib import Path
import pandas as pd

out = Path("outputs/champion")

rows = [
    {
        "Task": "Classification",
        "V1_Model": "Random Forest",
        "Final_Model": "Random Forest",
        "Primary_Metric": "F1",
        "V1_Value": 0.7010,
        "Final_Value": 0.7010,
        "Result": "V1 retained after rigorous V2 comparison",
    },
    {
        "Task": "Clustering",
        "V1_Model": "K-Means, 6 features",
        "Final_Model": "K-Means, RFM, K=2",
        "Primary_Metric": "Silhouette",
        "V1_Value": 0.3640,
        "Final_Value": 0.4386,
        "Result": "+20.50% relative improvement",
    },
    {
        "Task": "Sales forecasting",
        "V1_Model": "Random Forest",
        "Final_Model": "Extra Trees + calendar rule",
        "Primary_Metric": "MAE",
        "V1_Value": 10761.00,
        "Final_Value": 9576.10,
        "Result": "11.01% lower MAE",
    },
]

df = pd.DataFrame(rows)

df.to_csv(
    out / "FINAL_MODEL_SUMMARY.csv",
    index=False,
)

print(df.to_string(index=False))
