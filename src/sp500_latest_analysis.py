import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


# ============================
# 1) 시장 데이터 다운로드
# ============================
def load_market_data():
    """
    S&P500, VIX, 미국 10년 국채금리 데이터를 1년 치 받아온다.
    """
    end = datetime.today()
    start = end - timedelta(days=365)

    spx = yf.download("^GSPC", start=start, end=end, auto_adjust=False)
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=False)
    ust10 = yf.download("^TNX", start=start, end=end, auto_adjust=False)

    return spx, vix, ust10


# ============================
# 2) 시장 국면 판별 함수
# ============================
def detect_market_regime(spx: pd.DataFrame, vix: pd.DataFrame, ust10: pd.DataFrame):
    """
    이동평균(20/60/120), VIX, 금리 수준을 통해
    대략적인 시장 국면(상승장/하락장/횡보장)을 판별한다.
    """

    spx = spx.copy()

    # --- 0) Close 체크 / 보정 ---
    if "Close" not in spx.columns:
        if "Adj Close" in spx.columns:
            spx["Close"] = spx["Adj Close"]
        else:
            raise ValueError("S&P500 데이터에 Close/Adj Close 컬럼이 없습니다.")

    # --- 1) 이동평균 생성 ---
    spx["MA20"] = spx["Close"].rolling(20).mean()
    spx["MA60"] = spx["Close"].rolling(60).mean()
    spx["MA120"] = spx["Close"].rolling(120).mean()

    # --- 2) 이동평균 유효 행 필터 ---
    mask_valid = (
        spx["MA20"].notna() &
        spx["MA60"].notna() &
        spx["MA120"].notna()
    )
    if not mask_valid.any():
        raise ValueError("유효한 이동평균 데이터가 없습니다. (120거래일 이상 필요)")

    # 유효한 마지막 인덱스 하나 뽑기
    last_idx = spx.index[mask_valid][-1]

    # .at 으로 해당 행의 값을 '숫자'로 강제 추출 (Series 아님)
    ma20 = float(spx.at[last_idx, "MA20"])
    ma60 = float(spx.at[last_idx, "MA60"])
    ma120 = float(spx.at[last_idx, "MA120"])

    # --- 3) VIX / 금리 마지막 값 ---
    # VIX
    if "Close" in vix.columns:
        vix_close_series = vix["Close"]
    elif "Adj Close" in vix.columns:
        vix_close_series = vix["Adj Close"]
    else:
        raise ValueError(f"VIX 데이터에 Close/Adj Close가 없습니다. cols={vix.columns.tolist()}")

    vix_valid = vix_close_series.dropna()
    if vix_valid.empty:
        raise ValueError("VIX 종가 데이터가 없습니다.")
    vix_last = float(vix_valid.iloc[-1])

    # 10년물 금리 (^TNX)
    if "Close" in ust10.columns:
        ust_close_series = ust10["Close"]
    elif "Adj Close" in ust10.columns:
        ust_close_series = ust10["Adj Close"]
    else:
        raise ValueError(f"10년물 데이터에 Close/Adj Close가 없습니다. cols={ust10.columns.tolist()}")

    ust_valid = ust_close_series.dropna()
    if ust_valid.empty:
        raise ValueError("10년물 금리 종가 데이터가 없습니다.")
    rate_last = float(ust_valid.iloc[-1])  # ^TNX 는 보통 10배 값 (예: 41.2 ≈ 4.12%)

    # -----------------------
    # 4) 가격 추세 기반 판별
    # -----------------------
    if ma20 > ma60 and ma60 > ma120:
        trend = "강한 상승장"
    elif ma20 < ma60 and ma60 < ma120:
        trend = "하락장"
    else:
        trend = "횡보장"

    # -----------------------
    # 5) VIX 기반 보조판별
    # -----------------------
    if vix_last < 15:
        fear = "안정(상승 우위)"
    elif 15 <= vix_last <= 22:
        fear = "중립(횡보)"
    elif vix_last > 25:
        fear = "공포(하락 압력)"
    else:
        fear = "다소 불안"

    # -----------------------
    # 6) 금리 기반 보조판별 (^TNX는 10배 값)
    # -----------------------
    if rate_last < 40:       # 4% 미만
        rate_trend = "금리 안정 → 상승장 우호"
    elif 40 <= rate_last <= 45:
        rate_trend = "중립"
    else:
        rate_trend = "고금리 압박 → 하락장 우호"

    return {
        "trend": trend,
        "vix": vix_last,
        "fear": fear,
        "rate": rate_last,
        "rate_trend": rate_trend,
    }


# ============================
# 3) 전략 추천 엔진
# ============================
def recommend_strategy(regime: dict):
    print("\n===== 시장 국면 분석 결과 =====")
    print(f"📌 시장 추세: {regime['trend']}")
    print(f"📌 VIX 상태: {regime['vix']:.2f} → {regime['fear']}")
    print(f"📌 금리 상황: {regime['rate']:.2f} → {regime['rate_trend']}")

    print("\n===== 투자 전략 추천 =====")

    # 강한 상승장 + 낮은 VIX → 모멘텀
    if regime["trend"] == "강한 상승장" and regime["vix"] < 20:
        print("▶ 강한 상승장입니다 → **모멘텀 전략(Cluster 1)** 추천")
        return "momentum"

    # 횡보장 → 회복군
    if regime["trend"] == "횡보장":
        print("▶ 횡보장입니다 → **회복군(Reversal)** 전략 추천")
        return "recovery"

    # 하락장 또는 공포장 → 회복군
    if regime["trend"] == "하락장" or "공포" in regime["fear"]:
        print("▶ 하락/공포 국면입니다 → **회복군(Reversal)** 전략 추천")
        return "recovery"

    # 기본값: 회복군 약우위
    print("▶ 뚜렷하지 않은 국면입니다 → 회복군(Reversal)에 약한 우위")
    return "recovery"


# ============================
# 4) 전체 실행
# ============================
if __name__ == "__main__":
    print("시장 데이터를 불러오는 중...\n")
    spx, vix, ust10 = load_market_data()

    regime = detect_market_regime(spx, vix, ust10)
    strategy = recommend_strategy(regime)

    print("\n===== 최종 추천 전략 =====")
    if strategy == "momentum":
        print("📈 → 지금은 **모멘텀 전략(Cluster 1)**이 유리한 시장입니다.")
    else:
        print("🔄 → 지금은 **회복군 전략(Cluster 0, early<0 → mid/late>0)**이 유리한 시장입니다.")
