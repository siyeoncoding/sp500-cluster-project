import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from util_features import load_features_and_clusters

# ============================
# 0) 기본 설정
# ============================
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


# ============================
# 1) 데이터 + 군집 결과 불러오기
# ============================
raw_df, features_df, closing_cols_sorted = load_features_and_clusters()

print("=== 전체 종목 수 ===")
print(len(features_df))

print("\n=== 군집별 종목 수 ===")
print(features_df["cluster"].value_counts().sort_index())

# cluster0 여부 마스크만 먼저 만들어두고,
# 실제 cluster0_df는 나중에 (early/mid/late 계산 후) 만들자
cluster0_mask = (features_df["cluster"] == 0)


# ============================
# 2) 날짜를 3구간(초기/중기/후기)으로 나누기
#    → early / mid / late 수익률 계산
# ============================
closing_df = raw_df[closing_cols_sorted]
n_cols = len(closing_cols_sorted)

early_cols = closing_cols_sorted[: int(n_cols * 1/3)]
mid_cols   = closing_cols_sorted[int(n_cols * 1/3): int(n_cols * 2/3)]
late_cols  = closing_cols_sorted[int(n_cols * 2/3):]


def stage_return(col_list):
    """
    특정 기간(col_list)에 대해:
    첫날 종가 대비 마지막날 종가 수익률 계산
    """
    df_stage = raw_df[col_list]
    first = df_stage.iloc[:, 0]
    last = df_stage.iloc[:, -1]
    return last / first - 1


# features_df에 단계별 수익률 컬럼 추가
features_df["early_return"] = stage_return(early_cols)
features_df["mid_return"]   = stage_return(mid_cols)
features_df["late_return"]  = stage_return(late_cols)

print("\n=== 단계별 수익률 컬럼 추가 완료 ===")
print(features_df[["early_return", "mid_return", "late_return"]].head())

# 이제야 cluster0_df를 만든다 (early/mid/late 포함된 상태)
cluster0_df = features_df[cluster0_mask].copy()
print("\n=== Cluster 0 종목 수 ===")
print(len(cluster0_df))


# ============================
# 3) '회복 신호' 조건 정의
#    - 전체적으로는 Cluster 0 (저수익·중위험 군집)
#    - early_return < 0  : 초반에는 하락
#    - mid_return   > 0  : 중반부터 반등 시작
#    - late_return  > mid_return & late_return > 0 : 후반에 더 강한 상승
# ============================
recovery_mask = (
    (features_df["cluster"] == 0) &
    (features_df["early_return"] < 0) &
    (features_df["mid_return"] > 0) &
    (features_df["late_return"] > features_df["mid_return"]) &
    (features_df["late_return"] > 0)
)

features_df["is_recovery"] = recovery_mask

cluster0_df = features_df[cluster0_mask].copy()

recovery_df = features_df[recovery_mask].copy()
non_recovery_c0_df = features_df[cluster0_mask & (~recovery_mask)].copy()

print("\n=== 회복 신호 종목 수 (Cluster 0 내부) ===")
print(len(recovery_df))

print("\n=== 회복 신호 종목 예시 ===")
print(
    recovery_df[
        ["company_name", "ticker",
         "return_6m", "early_return", "mid_return", "late_return"]
    ].head(20)
)


# ============================
# 4) 회복 후보군 CSV 저장
# ============================
out_csv = "../data/recovery_candidates.csv"
recovery_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\n[CSV 저장 완료] 회복 후보군 리스트: {out_csv}")


# ============================
# 5) 그룹별 통계 비교
#    - Cluster 0 전체
#    - Cluster 0 비회복군
#    - 회복군(recovery)
#    - Cluster 1, Cluster 3 (비교용)
# ============================
cluster1_df = features_df[features_df["cluster"] == 1].copy()
cluster3_df = features_df[features_df["cluster"] == 3].copy()


def summary_stats(name, df):
    print(f"\n=== {name} 통계 ===")
    print(
        df[
            ["return_6m", "volatility",
             "early_return", "mid_return", "late_return"]
        ].describe()
    )


summary_stats("Cluster 0 전체", cluster0_df)
summary_stats("Cluster 0 비회복군", non_recovery_c0_df)
summary_stats("회복군 (Cluster 0 중 회복 신호)", recovery_df)
summary_stats("Cluster 1 (중수익·저위험군)", cluster1_df)
summary_stats("Cluster 3 (고수익·고위험군)", cluster3_df)


# ============================
# 6) 그래프 1:
#    그룹별 6개월 누적 수익률 평균 비교
# ============================
group_labels = []
group_returns = []


def add_group(label, df):
    if len(df) > 0:
        group_labels.append(label)
        group_returns.append(df["return_6m"].mean())


add_group("C0 전체", cluster0_df)
add_group("C0 비회복", non_recovery_c0_df)
add_group("C0 회복군", recovery_df)
add_group("C1", cluster1_df)
add_group("C3", cluster3_df)

plt.figure(figsize=(8, 6))
sns.barplot(x=group_labels, y=group_returns)
plt.title("군집/그룹별 6개월 누적 수익률 평균 비교")
plt.xlabel("그룹")
plt.ylabel("평균 6개월 수익률")
save_and_show("group_mean_return_comparison.png")


# ============================
# 7) 그래프 2:
#    Cluster 0 내부에서
#    회복군 vs 비회복군의 단계별 수익률 패턴
# ============================
def mean_stage_returns(df):
    return [
        df["early_return"].mean(),
        df["mid_return"].mean(),
        df["late_return"].mean()
    ]


stages = ["초기(early)", "중기(mid)", "후기(late)"]

c0_non = mean_stage_returns(non_recovery_c0_df)
c0_rec = mean_stage_returns(recovery_df)

plt.figure(figsize=(8, 6))
plt.plot(stages, c0_non, marker="o", label="C0 비회복군")
plt.plot(stages, c0_rec, marker="o", label="C0 회복군")
plt.title("Cluster 0 내 회복군 vs 비회복군 단계별 수익률 패턴")
plt.xlabel("기간 구간")
plt.ylabel("평균 구간 수익률")
plt.legend()
save_and_show("cluster0_recovery_vs_nonrecovery_stage_returns.png")

# 큰 의미가 없어보여
# # ============================
# # 8) 그래프 3:
# #    Cluster 0에서
# #    변동성 vs 6개월 수익률 (회복군 색 다르게)
# # ============================
# 
# # 회복 여부를 사람이 보기에 쉬운 문자열 라벨로 변환
# cluster0_df["recovery_label"] = np.where(
#     cluster0_df["is_recovery"],
#     "회복",
#     "비회복"
# )
# 
# plt.figure(figsize=(8, 6))
# sns.scatterplot(
#     data=cluster0_df,
#     x="volatility",
#     y="return_6m",
#     hue="recovery_label",
#     palette={  # 색깔 확실히 다르게 지정
#         "비회복": "#FF7F0E",   # 주황
#         "회복": "#1F77B4"      # 파랑
#     },
#     alpha=0.8
# )
# plt.title("Cluster 0 내 변동성 vs 6개월 수익률 (회복군 강조)")
# plt.xlabel("Volatility")
# plt.ylabel("6개월 수익률")
# plt.legend(title="회복 신호 여부")
# save_and_show("cluster0_volatility_vs_return_recovery_highlight.png")


# 각 그룹에 group 라벨 붙이기
box_c0_non = non_recovery_c0_df.copy()
box_c0_non["group"] = "C0 비회복"

box_c0_rec = recovery_df.copy()
box_c0_rec["group"] = "C0 회복군"

box_c1 = cluster1_df.copy()
box_c1["group"] = "C1"

box_c3 = cluster3_df.copy()
box_c3["group"] = "C3"

box_df = pd.concat([box_c0_non, box_c0_rec, box_c1, box_c3], ignore_index=True)

plt.figure(figsize=(9, 6))
sns.boxplot(
    data=box_df,
    x="group",
    y="return_6m"
)
plt.title("그룹별 6개월 수익률 분포 비교 (Boxplot)")
plt.xlabel("그룹")
plt.ylabel("6개월 수익률")
save_and_show("group_return_boxplot.png")


# ============================
# 9) 그래프 4:
#    그룹별 '플러스 수익률 종목 비율' 비교
# ============================

def positive_ratio(df):
    # 수익률이 0보다 큰 종목의 비율
    return (df["return_6m"] > 0).mean()

prob_labels = []
prob_values = []

def add_prob_group(label, df):
    if len(df) > 0:
        prob_labels.append(label)
        prob_values.append(positive_ratio(df))

add_prob_group("C0 전체", cluster0_df)
add_prob_group("C0 비회복", non_recovery_c0_df)
add_prob_group("C0 회복군", recovery_df)
add_prob_group("C1", cluster1_df)
add_prob_group("C3", cluster3_df)

plt.figure(figsize=(9, 6))
sns.barplot(x=prob_labels, y=prob_values)
plt.ylim(0, 1)
plt.title("그룹별 '플러스(>0) 6개월 수익률' 종목 비율 비교")
plt.xlabel("그룹")
plt.ylabel("플러스 수익률 비율 (확률)")
save_and_show("group_positive_return_probability.png")
