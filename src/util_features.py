# src/utils_features.py 로 만들어두면 더 좋음

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def load_features_and_clusters(csv_path="../data/sp500_2025_h1.csv",
                               n_clusters=4,
                               random_state=42):
    df = pd.read_csv(csv_path)

    # ---- 날짜별 컬럼 정리 ----
    closing_cols = [c for c in df.columns if c.endswith("_closing")]
    volume_cols = [c for c in df.columns if c.endswith("_volume")]

    def extract_date(col, suffix):
        return pd.to_datetime(col.replace(f"_{suffix}", ""), format="%d-%m-%Y")

    closing_cols_sorted = sorted(closing_cols, key=lambda c: extract_date(c, "closing"))
    volume_cols_sorted = sorted(volume_cols, key=lambda c: extract_date(c, "volume"))

    closing_df = df[closing_cols_sorted]
    volume_df = df[volume_cols_sorted]

    # ---- 파생 변수 ----
    return_6m = closing_df.iloc[:, -1] / closing_df.iloc[:, 0] - 1
    volatility = closing_df.pct_change(axis=1).std(axis=1)
    avg_volume = volume_df.mean(axis=1)
    up_ratio = (closing_df.diff(axis=1) > 0).sum(axis=1) / closing_df.diff(axis=1).notna().sum(axis=1)

    features_df = pd.DataFrame({
        "company_name": df["company_name"],
        "ticker": df["ticker"],
        "return_6m": return_6m,
        "volatility": volatility,
        "avg_volume": avg_volume,
        "up_ratio": up_ratio
    })

    # ---- KMeans로 클러스터 다시 학습 ----
    X = features_df[["return_6m", "volatility", "avg_volume", "up_ratio"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    features_df["cluster"] = kmeans.fit_predict(X_scaled)

    return df, features_df, closing_cols_sorted
