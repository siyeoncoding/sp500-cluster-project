import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use("ggplot")

# ===== 그래프 저장 폴더 =====
save_dir = "../figures"
os.makedirs(save_dir, exist_ok=True)

def save_and_show(filename):
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"저장 완료: {path}")
    plt.show()


# ===========================
# 1) 데이터 로드
# ===========================
file_path = "../data/sp500_2025_h1.csv"
df = pd.read_csv(file_path)

# === EDA 때 만든 features_df 다시 계산 ===
closing_cols = [c for c in df.columns if c.endswith("_closing")]
volume_cols = [c for c in df.columns if c.endswith("_volume")]

def extract_date(col, suffix):
    return pd.to_datetime(col.replace(f"_{suffix}", ""), format="%d-%m-%Y")

closing_cols_sorted = sorted(closing_cols, key=lambda c: extract_date(c, "closing"))
volume_cols_sorted = sorted(volume_cols, key=lambda c: extract_date(c, "volume"))

closing_df = df[closing_cols_sorted]
volume_df = df[volume_cols_sorted]

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

# ===========================
# 2) K-Means에 사용할 변수 선택
# ===========================
X = features_df[["return_6m", "volatility", "avg_volume", "up_ratio"]]

# ===========================
# 3) 표준화
# ===========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===========================
# 4) Elbow Method
# ===========================
sse = []
K_list = range(2, 10)

for k in K_list:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_list, sse, marker="o")
plt.title("Elbow Method (SSE)")
plt.xlabel("k")
plt.ylabel("SSE")
save_and_show("kmeans_elbow.png")

# ===========================
# 5) Silhouette Score 평가
# ===========================
sil_scores = []
for k in K_list:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(8,5))
plt.plot(K_list, sil_scores, marker="o")
plt.title("Silhouette Score")
plt.xlabel("k")
plt.ylabel("Score")
save_and_show("kmeans_silhouette.png")

# ===========================
# 6) 최종 k 선택 (예: 3)
# ===========================
optimal_k = sil_scores.index(max(sil_scores)) + 2
print("최적 k =", optimal_k)

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

features_df["cluster"] = cluster_labels

# ===========================
# 7) 군집별 분석
# ===========================
cluster_summary = features_df.groupby("cluster")[["return_6m", "volatility", "avg_volume", "up_ratio"]].mean()
print("\n=== 군집별 평균 값 ===")
print(cluster_summary)

cluster_summary.to_csv("../figures/cluster_summary.csv")
print("군집 결과 저장 완료: cluster_summary.csv")

# ===========================
# 8) 군집 시각화 (수익률 vs 변동성)
# ===========================
plt.figure(figsize=(10,7))
sns.scatterplot(
    data=features_df,
    x="volatility",
    y="return_6m",
    hue="cluster",
    palette="Set2",
    s=60
)
plt.title("K-Means 군집 결과: 수익률 vs 변동성")
plt.xlabel("Volatility")
plt.ylabel("6-month Return")
save_and_show("cluster_return_vs_volatility.png")
