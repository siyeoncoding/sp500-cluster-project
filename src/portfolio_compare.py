# src/portfolio_compare.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# 1) 데이터 + 클러스터 불러오기
raw_df, features_df, closing_cols_sorted = load_features_and_clusters()

closing_df = raw_df[closing_cols_sorted]

# 2) 일별 개별 종목 수익률 계산
#    (열 방향: 날짜, 행: 종목)
price = closing_df.values  # (503, 122)
daily_ret = price[:, 1:] / price[:, :-1] - 1  # (503, 121)

dates = [col for col in closing_cols_sorted[1:]]

# 3) 전체 포트폴리오 vs Cluster0 제외 포트폴리오
mask_all = np.ones(price.shape[0], dtype=bool)
mask_no_c0 = features_df["cluster"] != 0

ret_all = daily_ret[mask_all].mean(axis=0)
ret_no_c0 = daily_ret[mask_no_c0].mean(axis=0)

# 4) 누적 수익률 곡선 계산
cum_all = (1 + ret_all).cumprod()
cum_no_c0 = (1 + ret_no_c0).cumprod()

# 5) 그래프
plt.figure(figsize=(10,6))
plt.plot(dates, cum_all, label="전체 포트폴리오 (Equal-weight)")
plt.plot(dates, cum_no_c0, label="Cluster 0 제외 포트폴리오", linestyle="--")
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("누적 수익률 (기준=1)")
plt.title("전체 vs Cluster 0 제외 포트폴리오 누적 수익률 비교")
plt.legend()
save_and_show("portfolio_all_vs_noC0.png")

# 6) 요약 통계 출력
def summarize(ret_series, name):
    mean_ret = ret_series.mean()
    vol = ret_series.std()
    sharpe = mean_ret / vol if vol != 0 else np.nan
    print(f"\n[{name}]")
    print("평균 일간 수익률:", mean_ret)
    print("일간 변동성:", vol)
    print("Sharpe 비율(단순):", sharpe)

summarize(ret_all, "전체 포트폴리오")
summarize(ret_no_c0, "Cluster 0 제외 포트폴리오")
