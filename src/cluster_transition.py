# src/cluster_transition.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use("ggplot")

save_dir = "../figures"
os.makedirs(save_dir, exist_ok=True)

def save_and_show(name):
    path = os.path.join(save_dir, name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print("저장:", path)
    plt.show()

# 1) 원본 로드
file_path = "../data/sp500_2025_h1.csv"
df = pd.read_csv(file_path)

# 날짜 파싱 함수
def extract_date(col, suffix):
    return pd.to_datetime(col.replace(f"_{suffix}", ""), format="%d-%m-%Y")

closing_cols = [c for c in df.columns if c.endswith("_closing")]
volume_cols = [c for c in df.columns if c.endswith("_volume")]

closing_cols_sorted = sorted(closing_cols, key=lambda c: extract_date(c, "closing"))
volume_cols_sorted = sorted(volume_cols, key=lambda c: extract_date(c, "volume"))

closing_df_all = df[closing_cols_sorted]
volume_df_all = df[volume_cols_sorted]

dates = [extract_date(c, "closing") for c in closing_cols_sorted]
date_series = pd.Series(dates, index=closing_cols_sorted)

# 2) 기간 정의 (연속된 월 기준으로 크게 3개 구간)
period_defs = {
    "P1_1-2월": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-28")),
    "P2_3-4월": (pd.Timestamp("2025-03-01"), pd.Timestamp("2025-04-30")),
    "P3_5-6월": (pd.Timestamp("2025-05-01"), pd.Timestamp("2025-06-30")),
}

period_labels = list(period_defs.keys())
cluster_results = pd.DataFrame({"ticker": df["ticker"]})

# 3) 각 기간별로 클러스터링
for label, (start, end) in period_defs.items():
    period_cols_close = [c for c in closing_cols_sorted
                         if start <= date_series[c] <= end]
    period_cols_vol = [c for c in volume_cols_sorted
                       if start <= extract_date(c, "volume") <= end]

    closing_df = df[period_cols_close]
    volume_df = df[period_cols_vol]

    # 기간 수익률/변동성 등 계산
    ret = closing_df.iloc[:, -1] / closing_df.iloc[:, 0] - 1
    vol = closing_df.pct_change(axis=1).std(axis=1)
    avg_vol = volume_df.mean(axis=1)
    up_ratio = (closing_df.diff(axis=1) > 0).sum(axis=1) / closing_df.diff(axis=1).notna().sum(axis=1)

    feats = pd.DataFrame({
        "return": ret,
        "volatility": vol,
        "avg_volume": avg_vol,
        "up_ratio": up_ratio
    })

    X_scaled = StandardScaler().fit_transform(feats)
    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    cluster_results[label] = cluster_labels

# 4) P1 → P2, P2 → P3 전이 행렬(heatmap)
def plot_transition_matrix(from_col, to_col, title, filename):
    trans = pd.crosstab(cluster_results[from_col], cluster_results[to_col])
    plt.figure(figsize=(6,5))
    sns.heatmap(trans, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel(to_col + " (도착 클러스터)")
    plt.ylabel(from_col + " (출발 클러스터)")
    save_and_show(filename)

plot_transition_matrix("P1_1-2월", "P2_3-4월",
                       "클러스터 전이: 1–2월 → 3–4월",
                       "transition_P1_P2.png")

plot_transition_matrix("P2_3-4월", "P3_5-6월",
                       "클러스터 전이: 3–4월 → 5–6월",
                       "transition_P2_P3.png")
