import pandas as pd

def detect_cpu_spikes(df: pd.DataFrame, threshold: Float = 80.0) -> pd.DataFrame:
    return df[df["cpu"]>threshold]

def group_errors_by_service(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("service")["error_type"].count().sort_values(ascending=False)

#Create Dummy DataFrame
data = {
    "timestamp": pd.date_range("2025-01-01 03:00", periods=10, freq="min"),
    "service": ["auth", "db", "auth", "db", "api", "api", "auth", "db", "api", "auth"],
    "cpu": [50, 92, 87, 45, 99, 30, 82, 70, 88, 60],
    "error_type": ["Timeout", "ConnError", "Timeout", "None", "OOM", "None", "Timeout", "None", "OOM", "None"]
}
df=pd.DataFrame(data)