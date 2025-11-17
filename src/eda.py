import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ===== 한글 폰트 설정 =====
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use("ggplot")

# ===== 그래프 저장 폴더 생성 =====
save_dir = "../figures"
os.makedirs(save_dir, exist_ok=True)

def save_and_show(filename):
    """그래프 저장 후 화면에 출력"""
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"저장 완료: {path}")
    plt.show()


# ===========================
# 1) 데이터 불러오기
# ===========================
file_path = "../data/sp500_2025_h1.csv"
df = pd.read_csv(file_path)

# ===========================
# 2) 메타 & 날짜별 컬럼 분리
# ===========================
meta_cols = ["company_name", "ticker"]
closing_cols = [c for c in df.columns if c.endswith("_closing")]
volume_cols = [c for c in df.columns if c.endswith("_volume")]

# ===========================
# 3) 날짜 정렬
# ===========================
def extract_date(col, suffix):
    date_str = col.replace(f"_{suffix}", "")
    return pd.to_datetime(date_str, format="%d-%m-%Y")

closing_cols_sorted = sorted(closing_cols, key=lambda c: extract_date(c, "closing"))
volume_cols_sorted = sorted(volume_cols, key=lambda c: extract_date(c, "volume"))

closing_df = df[closing_cols_sorted]
volume_df = df[volume_cols_sorted]

# ===========================
# 4) 파생 변수 계산
# ===========================
return_6m = closing_df.iloc[:, -1] / closing_df.iloc[:, 0] - 1
volatility = closing_df.pct_change(axis=1).std(axis=1)
avg_volume = volume_df.mean(axis=1)
up_ratio = (closing_df.diff(axis=1) > 0).sum(axis=1) / closing_df.diff(axis=1).notna().sum(axis=1)

# ===========================
# 5) Feature DF 생성
# ===========================
features_df = pd.DataFrame({
    "company_name": df["company_name"],
    "ticker": df["ticker"],
    "return_6m": return_6m,
    "volatility": volatility,
    "avg_volume": avg_volume,
    "up_ratio": up_ratio
})


# ===========================
# 6) 그래프 시각화 + 저장
# ===========================

# ---- 그래프 1: 수익률 분포 ----
plt.figure(figsize=(10,6))
sns.histplot(features_df["return_6m"], bins=40, kde=True)
plt.title("S&P 500 6개월 누적 수익률 분포")
plt.xlabel("Return (6 months)")
plt.ylabel("Count")
save_and_show("return_distribution.png")


# ---- 그래프 2: 변동성 분포 ----
plt.figure(figsize=(10,6))
sns.histplot(features_df["volatility"], bins=40, kde=True)
plt.title("S&P 500 주식 변동성 분포")
plt.xlabel("Volatility")
plt.ylabel("Count")
save_and_show("volatility_distribution.png")


# ---- 그래프 3: 평균 거래량 분포 ----
plt.figure(figsize=(10,6))
sns.histplot(features_df["avg_volume"], bins=40)
plt.title("S&P 500 평균 거래량 분포")
plt.xlabel("Average Volume")
plt.ylabel("Count")
save_and_show("avg_volume_distribution.png")


# ---- 그래프 4: 상승일 비율 분포 ----
plt.figure(figsize=(10,6))
sns.histplot(features_df["up_ratio"], bins=30)
plt.title("S&P 500 상승일 비율 분포")
plt.xlabel("Up Ratio")
plt.ylabel("Count")
save_and_show("up_ratio_distribution.png")


# ---- 그래프 5: 수익률 vs 변동성 ----
plt.figure(figsize=(10,7))
sns.scatterplot(data=features_df, x="volatility", y="return_6m", alpha=0.7)
plt.title("수익률 vs 변동성 (위험-수익 플롯)")
plt.xlabel("Volatility")
plt.ylabel("6-month Return")
save_and_show("return_vs_volatility.png")


# ---- 그래프 6: 수익률 vs 평균 거래량 ----
plt.figure(figsize=(10,7))
sns.scatterplot(data=features_df, x="avg_volume", y="return_6m", alpha=0.7)
plt.title("수익률 vs 평균 거래량")
plt.xlabel("Average Volume")
plt.ylabel("6-month Return")
save_and_show("return_vs_avg_volume.png")


# ---- 그래프 7: 변동성 vs 평균 거래량 ----
plt.figure(figsize=(10,7))
sns.scatterplot(data=features_df, x="avg_volume", y="volatility", alpha=0.7)
plt.title("변동성 vs 평균 거래량")
plt.xlabel("Average Volume")
plt.ylabel("Volatility")
save_and_show("volatility_vs_avg_volume.png")
