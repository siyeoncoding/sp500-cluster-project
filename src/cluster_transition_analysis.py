import pandas as pd
from util_features import load_features_and_clusters

# 전체 데이터 + 군집 결과
raw_df, features_df, closing_cols_sorted = load_features_and_clusters()

# 첫 2개월(1~2월), 중간 2개월(3~4월), 마지막 2개월(5~6월) 종가만 분리
closing_df = raw_df[closing_cols_sorted]

# 3단계 기간 나누기
n = len(closing_cols_sorted)
early = closing_cols_sorted[:int(n*0.33)]
mid   = closing_cols_sorted[int(n*0.33):int(n*0.66)]
late  = closing_cols_sorted[int(n*0.66):]

def get_stage_return(cols):
    df = raw_df[cols]
    return df.iloc[:, -1] / df.iloc[:, 0] - 1

features_df["early_return"] = get_stage_return(early)
features_df["mid_return"] = get_stage_return(mid)
features_df["late_return"] = get_stage_return(late)

# Cluster 0 → Cluster 1 수준으로 성과가 개선된 종목 찾기
transition_candidates = features_df[
    (features_df["cluster"] == 0) &
    (features_df["early_return"] < 0) &         # 초반 약세
    (features_df["mid_return"] > 0) &           # 중반 개선
    (features_df["late_return"] > features_df["mid_return"])  # 후반 추세 강화
]

transition_candidates.to_csv("../data/cluster_transition_candidates.csv",
                             index=False, encoding="utf-8-sig")

print("=== 회복 신호가 있는 Cluster 0 종목 ===")
print(transition_candidates[["company_name", "ticker",
                             "early_return", "mid_return", "late_return"]])
