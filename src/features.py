import pandas as pd
import numpy as np

# 1) CSV 파일 경로
file_path = "../data/sp500_2025_h1.csv"

# 2) CSV 읽기
df = pd.read_csv(file_path)

print("원본 데이터 크기:", df.shape)
print("컬럼 예시:", df.columns[:10])

# 메타 정보 컬럼 (회사 이름, 티커)
meta_cols = ["company_name", "ticker"]

# 종가, 시가, 거래량 컬럼 분리
closing_cols = [c for c in df.columns if c.endswith("_closing")]
opening_cols = [c for c in df.columns if c.endswith("_opening")]
volume_cols = [c for c in df.columns if c.endswith("_volume")]

print("종가 컬럼 개수:", len(closing_cols))
print("시가 컬럼 개수:", len(opening_cols))
print("거래량 컬럼 개수:", len(volume_cols))
print("종가 컬럼 예시:", closing_cols[:5])


# 문자열에서 날짜 부분만 떼서 datetime으로 변환 → 정렬 기준으로 사용
def extract_date_from_col(col_name: str, suffix: str) -> pd.Timestamp:
    """
    예: "02-01-2025_closing" + suffix="closing"
        -> "02-01-2025" -> datetime(2025-01-02)
    """
    date_str = col_name.replace(f"_{suffix}", "")  # "_closing" 제거
    # 날짜 포맷은 '일-월-연도' 형식 (02-01-2025 = 2 Jan 2025로 가정)
    return pd.to_datetime(date_str, format="%d-%m-%Y")

# 종가 컬럼을 날짜 기준으로 정렬
closing_cols_sorted = sorted(
    closing_cols,
    key=lambda c: extract_date_from_col(c, "closing")
)

volume_cols_sorted = sorted(
    volume_cols,
    key=lambda c: extract_date_from_col(c, "volume")
)

print("정렬된 종가 컬럼 앞 5개:", closing_cols_sorted[:5])
print("정렬된 종가 컬럼 뒤 5개:", closing_cols_sorted[-5:])


# 종가만 모은 부분 DataFrame (회사 + 종가들)
closing_df = df[meta_cols + closing_cols_sorted]

# 각 회사의 첫 종가, 마지막 종가
first_close = closing_df[closing_cols_sorted[0]]
last_close = closing_df[closing_cols_sorted[-1]]

# 6개월 누적 수익률
return_6m = (last_close / first_close) - 1

print("=== 6개월 누적 수익률 예시(앞 5개) ===")
print(return_6m.head())


# 종가 부분만 숫자만 떼기 (meta 제외)
closing_values = closing_df[closing_cols_sorted]

# 일별 수익률: 각 회사(row)별로, 날짜(axis=1) 방향으로 퍼센트 변화
daily_returns = closing_values.pct_change(axis=1)

# 변동성 = 일별 수익률의 표준편차
volatility = daily_returns.std(axis=1, skipna=True)

print("=== 변동성 예시(앞 5개) ===")
print(volatility.head())


volume_df = df[meta_cols + volume_cols_sorted]
volume_values = volume_df[volume_cols_sorted]

avg_volume = volume_values.mean(axis=1)

print("=== 평균 거래량 예시(앞 5개) ===")
print(avg_volume.head())


# 종가 변화량 (오늘 - 어제), axis=1 방향으로 차분
close_diff = closing_values.diff(axis=1)

# 상승한 날: 오늘종가 - 어제종가 > 0
up_days = (close_diff > 0).sum(axis=1)

# 비교 가능한 날짜 수 (NaN 아닌 곳)
valid_days = close_diff.notna().sum(axis=1)

# 상승일 비율
up_ratio = up_days / valid_days

print("=== 상승일 비율 예시(앞 5개) ===")
print(up_ratio.head())


features_df = pd.DataFrame({
    "company_name": df["company_name"],
    "ticker": df["ticker"],
    "return_6m": return_6m,
    "volatility": volatility,
    "avg_volume": avg_volume,
    "up_ratio": up_ratio,
})

print("=== 요약 특성 DataFrame 크기 ===")
print(features_df.shape)

print("=== 요약 특성 앞 5행 ===")
print(features_df.head())
