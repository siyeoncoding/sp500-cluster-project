import pandas as pd
import numpy as np
import kagglehub
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from datetime import timedelta


# ==========================================
# 1) Kaggle Wide → Long 변환 함수
# ==========================================
def load_sp500_csv(csv_path: str) -> pd.DataFrame:
    """
    Kaggle의 SnP_daily_update.csv를 읽어서
    (Date, Ticker, Close, Volume) 형태의 long format으로 변환한다.
    - long format이면 그대로 사용
    - wide format이면 melt로 변환
    """
    df = pd.read_csv(csv_path, low_memory=False)
    cols = df.columns.tolist()

    long_format_cols = {"Date", "Ticker", "Close"}
    # 이미 Long 형식인 경우
    if long_format_cols.issubset(cols):
        print("Long format detected.")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # 여기까지 왔으면 wide 형식
    print("Wide format detected → converting to long format...")

    # 0번째 컬럼은 날짜로 가정 (Price 등 이름이지만)
    df = df.rename(columns={df.columns[0]: "Date"})

    # 1행이 Ticker header인지 자동 감지
    first_row = df.iloc[0].tolist()
    is_header_row = True
    for x in first_row[1:5]:
        if not isinstance(x, str):
            is_header_row = False
            break

    if is_header_row:
        print("Detected ticker header row → removing row 0")
        ticker_names = first_row[1:]  # 티커 리스트
        df = df.iloc[1:].reset_index(drop=True)
    else:
        # 컬럼 이름이 바로 티커면 이걸 사용
        ticker_names = df.columns[1:]

    # 날짜 변환
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # melt → long format
    value_cols = df.columns[1:]
    df_long = df.melt(
        id_vars="Date",
        value_vars=value_cols,
        var_name="Ticker",
        value_name="PriceValue"   # 임시 이름 (Close와 충돌 방지)
    )

    # header row에서 읽은 티커로 덮어쓰기
    if is_header_row:
        print("Applying header-row ticker mapping...")
        # df는 (날짜 수) row, ticker_names는 (티커 수)
        # melt 후 row 수 = 날짜 수 * 티커 수 → repeat으로 매칭
        df_long["Ticker"] = np.repeat(ticker_names, len(df))

    # 숫자로 변환
    df_long["PriceValue"] = pd.to_numeric(df_long["PriceValue"], errors="coerce")
    df_long = df_long.dropna(subset=["PriceValue"])

    # Volume은 이 파일에 없으므로 NaN으로 채움
    df_long["Volume"] = np.nan

    # 최종 컬럼 이름 통일
    df_long = df_long.rename(columns={"PriceValue": "Close"})

    print("Wide → Long 변환 완료. Row count:", len(df_long))
    return df_long.sort_values(["Ticker", "Date"]).reset_index(drop=True)


# ==========================================
# 2) 1년치 특징 생성 함수
# ==========================================
def make_features(df_long: pd.DataFrame, end_date: pd.Timestamp,
                  lookback_days: int = 365) -> pd.DataFrame:
    """
    end_date 기준으로 과거 lookback_days일(기본 1년)의 데이터를 사용해
    티커별 특징을 만든다.
    - return_6m: 1년 누적 수익률 (이름은 유지하지만 1년 기준)
    - volatility: 일별 수익률 표준편차
    - early/mid/late return: 1년을 3구간으로 나눈 구간 수익률
    """
    start_date = end_date - timedelta(days=lookback_days)
    df_win = df_long[(df_long["Date"] >= start_date) & (df_long["Date"] <= end_date)]
    print(f"{lookback_days}일(약 {lookback_days/30:.1f}개월) 구간 데이터 크기: {df_win.shape}")

    features = []
    for ticker, g in df_win.groupby("Ticker"):
        g = g.sort_values("Date")
        close = g["Close"]

        # 데이터가 너무 적으면 스킵
        if len(close) < 60:  # 대략 3개월 이상
            continue

        # 전체 구간 누적 수익률
        ret_total = close.iloc[-1] / close.iloc[0] - 1
        # 일일 수익률 기준 변동성
        vol = close.pct_change().std()

        n = len(g)
        e_end = int(n * (1/3))
        m_end = int(n * (2/3))

        early = close.iloc[:e_end]
        mid = close.iloc[e_end:m_end]
        late = close.iloc[m_end:]

        def seg_ret(seg):
            return seg.iloc[-1] / seg.iloc[0] - 1 if len(seg) > 1 else 0.0

        features.append({
            "Ticker": ticker,
            "return_6m": ret_total,   # 이름은 그대로 두되 실제로는 1년치
            "volatility": vol,
            "early_return": seg_ret(early),
            "mid_return": seg_ret(mid),
            "late_return": seg_ret(late),
        })

    return pd.DataFrame(features)


# ==========================================
# 3) KMeans 군집화 (K=4)
# ==========================================
def assign_clusters(df_feat: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    X = df_feat[["return_6m", "volatility", "early_return", "mid_return", "late_return"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_feat = df_feat.copy()
    df_feat["cluster"] = kmeans.fit_predict(X_scaled)
    return df_feat


# ==========================================
# 4) 회복 라벨 생성: 이전 1년 C0 → 최근 1년 C1
# ==========================================
def make_recovery_label(df_today: pd.DataFrame, df_prev: pd.DataFrame) -> pd.DataFrame:
    """
    df_prev: 이전 1년 군집 정보
    df_today: 최근 1년 군집 정보
    label:
      - prev_cluster == 0 and cluster == 1 → 1 (회복주)
      - else → 0
    """
    df_prev_idx = df_prev.set_index("Ticker")
    df_today = df_today.copy()

    df_today["prev_cluster"] = df_today["Ticker"].map(df_prev_idx["cluster"])
    df_today["label"] = np.where(
        (df_today["prev_cluster"] == 0) & (df_today["cluster"] == 1),
        1, 0
    )
    return df_today


# ==========================================
# 5) XGBoost 학습 & 예측
# ==========================================
def train_predict_xgb(df_labeled: pd.DataFrame):
    df = df_labeled.dropna(subset=["label"])
    X = df[["return_6m", "volatility", "early_return", "mid_return", "late_return"]]
    y = df["label"]

    vc = y.value_counts()
    print("\n=== 라벨 분포 (0=비회복, 1=회복) ===")
    print(vc)

    # 라벨이 한 종류뿐이면 학습 불가
    if len(vc) < 2:
        print("\n[경고] 라벨이 한 클래스(0 또는 1)만 존재합니다.")
        print("→ 이 기준(이전 1년 C0 → 최근 1년 C1)으로 정의된 회복주가 거의 없습니다.")
        print("  회복주 예측 모델(XGBoost)은 학습할 수 없으므로, pred_prob를 0으로 채웁니다.")
        df["pred_prob"] = 0.0
        return df, None

    # 한 클래스 샘플 수가 너무 적으면 stratify 사용 불가
    use_stratify = True
    if vc.min() < 2:
        print("\n[주의] 한 클래스의 샘플 수가 2개 미만입니다.")
        print("→ stratify 옵션을 제거하고 일반 train_test_split을 사용합니다.")
        use_stratify = False

    if use_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

    model = XGBClassifier(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    df = df.copy()
    df["pred_prob"] = model.predict_proba(X)[:, 1]
    return df, model


# ==========================================
# MAIN 실행
# ==========================================
if __name__ == "__main__":
    print("Kaggle 데이터셋 다운로드 중...")
    path = kagglehub.dataset_download("yash16jr/s-and-p500-daily-update-dataset")
    csv_path = os.path.join(path, "SnP_daily_update.csv")
    print("사용할 CSV:", csv_path)

    # 1) 원본 데이터 로드 (long format 변환 포함)
    df_raw = load_sp500_csv(csv_path)

    # 2) 최신 날짜 및 1년 전 기준 설정
    end_date = df_raw["Date"].max()
    prev_end_date = end_date - timedelta(days=365)

    print("최신 날짜:", end_date)
    print("1년 전 기준 날짜:", prev_end_date)

    # 3) 최근 1년 특징 & 군집
    print("\n[최근 1년 특징 생성]")
    df_today_feat = make_features(df_raw, end_date, lookback_days=365)
    df_today_feat = assign_clusters(df_today_feat, n_clusters=4)

    # 4) 이전 1년 특징 & 군집
    print("\n[이전 1년 특징 생성]")
    df_prev_feat = make_features(df_raw, prev_end_date, lookback_days=365)

    df_prev_feat = df_prev_feat.dropna()

    df_prev_feat = assign_clusters(df_prev_feat, n_clusters=4)

    # 5) 회복 라벨 생성 (이전 1년 C0 → 최근 1년 C1)
    print("\n[회복 라벨 생성: prev C0 → now C1]")
    df_labeled = make_recovery_label(df_today_feat, df_prev_feat)

    # 6) XGBoost 학습 & 예측
    print("\n[XGBoost 회복 예측 모델 학습]")
    df_pred, model = train_predict_xgb(df_labeled)

    # 7) TOP 20 회복 가능 종목 출력
    print("\n🔥 회복 가능성이 가장 높은 TOP 20 종목 (pred_prob 기준)")
    top20 = df_pred.sort_values("pred_prob", ascending=False).head(20)
    print(top20[["Ticker", "pred_prob", "cluster", "prev_cluster", "return_6m",
                 "early_return", "mid_return", "late_return"]])

    # 8) CSV 저장 (선택)
    out_dir = "../data_latest_1y"
    os.makedirs(out_dir, exist_ok=True)
    df_pred.to_csv(os.path.join(out_dir, "recovery_pred_1y.csv"), index=False)
    top20.to_csv(os.path.join(out_dir, "recovery_top20_1y.csv"), index=False)
    print(f"\n[CSV 저장 완료] 전체 결과: {os.path.join(out_dir, 'recovery_pred_1y.csv')}")
    print(f"[CSV 저장 완료] TOP20: {os.path.join(out_dir, 'recovery_top20_1y.csv')}")
