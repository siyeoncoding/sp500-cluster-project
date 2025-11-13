import pandas as pd

# 1) CSV 파일 경로 설정
#    - 지금 프로젝트 구조 기준으로, 실행 위치는 프로젝트 최상단(sp500_cluster_project)
#    - data 폴더 안에 csv가 있으니, 상대 경로는 "data/파일이름"
file_path = "data/sp500_2025_h1.csv"

# 2) CSV 파일 읽기
df = pd.read_csv(file_path)

# 3) 데이터 기본 정보 출력
print("===== 데이터 기본 정보 =====")
print("행(row) 수, 열(column) 수:", df.shape)   # (행 개수, 열 개수)

print("\n===== 앞 5행 미리 보기(head) =====")
print(df.head())   # 위에서부터 5행만 출력

print("\n===== 컬럼 이름 확인 =====")
print(df.columns.tolist())   # 컬럼 이름을 리스트로 출력

print("\n===== 각 컬럼별 데이터 타입(dtypes) =====")
print(df.dtypes.head(20))    # 앞쪽 20개 컬럼만 타입 확인

