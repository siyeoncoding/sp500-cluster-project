import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from util_features import load_features_and_clusters

# ======== 기본 설정 ========
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use("ggplot")

FIG_DIR = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)


def save_and_show(filename):
    """그래프 저장 + 표시"""
    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[그림 저장] {path}")
    plt.show()


# ======== 1. 전체 데이터 & 군집 불러오기 ========
raw_df, features_df, closing_cols_sorted = load_features_and_clusters()

print("=== 전체 종목 수 ===")
print(len(features_df))

print("\n=== 군집별 종목 수 ===")
print(features_df["cluster"].value_counts().sort_index())

# ======== 2. Cluster 0 필터링 ========
cluster0_df = features_df[features_df["cluster"] == 0].copy()

print("\n=== Cluster 0 종목 수 ===")
print(len(cluster0_df))

print("\n=== Cluster 0 기본 통계 (return_6m, volatility, avg_volume, up_ratio) ===")
print(cluster0_df[["return_6m", "volatility", "avg_volume", "up_ratio"]].describe())

# ======== 3. Cluster 0 전체 리스트 CSV 저장 ========
csv_path = "../data/cluster0_stocks.csv"
cluster0_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\n[CSV 저장 완료] Cluster 0 종목 리스트: {csv_path}")

print("\n=== Cluster 0 종목 예시 상위 10개 ===")
print(cluster0_df[["company_name", "ticker", "return_6m", "volatility"]].head(10))

# ======== 4. 수익률 최악 TOP 20 (피해야 할 대표 종목) ========
worst20 = cluster0_df.sort_values("return_6m").head(20)

print("\n=== Cluster 0 수익률 최악 TOP 20 ===")
print(worst20[["company_name", "ticker", "return_6m", "volatility"]])

worst20_path = "../data/cluster0_worst20.csv"
worst20.to_csv(worst20_path, index=False, encoding="utf-8-sig")
print(f"\n[CSV 저장 완료] Cluster 0 수익률 최악 TOP 20: {worst20_path}")

# ======== 5. (옵션) 섹터 분석: 매핑 파일이 있을 때만 실행 ========
sector_map_path = "../data/sp500_sector_mapping.csv"

if os.path.exists(sector_map_path):
    print(f"\n[섹터 매핑 사용] {sector_map_path}")
    sector_map = pd.read_csv(sector_map_path)

    # 전체 + Cluster0 에 섹터 정보 붙이기
    all_with_sector = features_df.merge(sector_map, on="ticker", how="left")
    c0_with_sector = cluster0_df.merge(sector_map, on="ticker", how="left")

    # --- 5-1) Cluster 0 섹터 분포 ---
    sector_counts_c0 = (
        c0_with_sector["sector"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "sector", "sector": "count"})
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=sector_counts_c0, x="sector", y="count")
    plt.title("Cluster 0 섹터 분포 (피해야 할 종목이 많이 포함된 섹터)")
    plt.xlabel("Sector")
    plt.ylabel("종목 수")
    plt.xticks(rotation=45, ha="right")
    save_and_show("cluster0_sector_distribution.png")

    # --- 5-2) 전체 vs Cluster0 섹터 비중 비교 ---
    sector_counts_all = (
        all_with_sector["sector"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "sector", "sector": "count_all"})
    )

    merged_sector = sector_counts_all.merge(
        sector_counts_c0, on="sector", how="left"
    ).fillna(0)

    merged_sector["ratio_all"] = (
        merged_sector["count_all"] / merged_sector["count_all"].sum()
    )
    merged_sector["ratio_c0"] = (
        merged_sector["count"] / merged_sector["count"].sum()
    )

    merged_sector = merged_sector.sort_values("ratio_c0", ascending=False)

    plt.figure(figsize=(10, 6))
    width = 0.35
    idx = range(len(merged_sector))

    plt.bar(
        [i - width / 2 for i in idx],
        merged_sector["ratio_all"],
        width=width,
        label="전체 섹터 비중",
    )
    plt.bar(
        [i + width / 2 for i in idx],
        merged_sector["ratio_c0"],
        width=width,
        label="Cluster 0 섹터 비중",
    )

    plt.xticks(idx, merged_sector["sector"], rotation=45, ha="right")
    plt.ylabel("비중")
    plt.title("전체 vs Cluster 0 섹터 비중 비교")
    plt.legend()
    save_and_show("cluster0_sector_ratio_compare.png")

    print("\n=== 전체 vs Cluster 0 섹터 비중 비교 테이블 ===")
    print(
        merged_sector[
            ["sector", "count_all", "count", "ratio_all", "ratio_c0"]
        ]
    )

else:
    print(
        f"\n[주의] 섹터 매핑 파일이 없어 섹터 분석을 건너뜁니다.\n"
        f"경로에 파일을 만들면 자동으로 분석됩니다: {sector_map_path}\n"
        f"(형식 예시: ticker,sector 로 구성된 CSV)"
    )

print("\n=== Cluster 0 분석 완료 ===")
