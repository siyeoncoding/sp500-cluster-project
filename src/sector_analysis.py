# src/sector_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from util_features import load_features_and_clusters

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

# 1) features + cluster 불러오기
raw_df, features_df, _ = load_features_and_clusters()

# 2) 섹터 매핑 불러와서 merge
sector_map = pd.read_csv("../data/sp500_sector_mapping.csv")  # ticker, sector
features_sector = features_df.merge(sector_map, on="ticker", how="left")

# 3) 섹터별 군집 비율 (stacked bar)
cluster_counts = (
    features_sector
    .groupby(["sector", "cluster"])
    .size()
    .reset_index(name="count")
)

total_per_sector = cluster_counts.groupby("sector")["count"].transform("sum")
cluster_counts["ratio"] = cluster_counts["count"] / total_per_sector

plt.figure(figsize=(12,6))
cluster_pivot = cluster_counts.pivot(index="sector", columns="cluster", values="ratio")
cluster_pivot.plot(kind="bar", stacked=True, figsize=(12,6), colormap="Set2")
plt.title("섹터별 군집 비율 (Cluster 0~3)")
plt.xlabel("Sector")
plt.ylabel("비율")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Cluster")
save_and_show("sector_cluster_ratio.png")

# 4) 섹터별 평균 수익률 (cluster별로 비교)
sector_return = (
    features_sector
    .groupby(["sector", "cluster"])["return_6m"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12,6))
sns.barplot(data=sector_return, x="sector", y="return_6m", hue="cluster")
plt.title("섹터별 · 군집별 평균 6개월 수익률")
plt.xlabel("Sector")
plt.ylabel("Return (6m)")
plt.xticks(rotation=45, ha="right")
save_and_show("sector_cluster_return.png")
