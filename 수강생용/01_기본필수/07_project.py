"""07~08 기본필수: 이 파일 하나를 완성하고 같은 파일로 재실행합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

HERE = Path(__file__).resolve().parent # 파일 경로 설정
data = pd.read_csv(HERE / "requests.csv") # requests.csv파일 읽어서 data에 대입

# 전체 데이터 중 80%는 train_valid, 20%는 test에 저장
train_valid, test = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["failed"])

# 17라인에서 처리된 데이터(train_valid, test)중 train_valid의 25%를 valid에 저장, 75프로는 train에 저장
train, valid = train_test_split(
    train_valid, test_size=0.25, random_state=42, stratify=train_valid["failed"])
#train:valid:test = 6:2:2 으로 나눠진 데이터
#train : 전처리 기준을 학습하고 모델을 훈련하는 데이터
#valid : 학습이 끝난 모델의 예측 성능을 확인하는 데이터
#test  : 최종 성능을 평가하기 위해 남겨두는 데이터

features = ["input_chars", "task_type"] # 입력 열 설정
X_train = train[features]#모델이 학습할 때 입력으로 활용하는 데이터
y_train = train["failed"]#학습용 정답 데이터
X_valid = valid[features]#학습이 끝난 모델에게 예측을 시킬 데이터

#열마다 서로 다른 전처리 방법 선택
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["input_chars"]), #input_chars 열에 숫자 데이터 표준화
    ("cat", OneHotEncoder(sparse_output=False), ["task_type"]), # task_type열의 문자 범주를 0과 1로 변환
])

# 파이프 라인 생성 : preprocessor로 task_type과 input_chars열 데이터 처리 후 분류 모델 생성 
pipeline = Pipeline([("pre", preprocessor),
                     ("model", DummyClassifier(strategy="most_frequent"))])
# 분류 모델 생성(학습 데이터에서 가장 자주 등장한 정답)

# TODO B07-1: pipeline.fit(X_train, y_train)으로 train에서만 학습하세요. pass를 지우고 작성합니다.
#학습용 정답 데이터(y_train)와 학습용 입력 데이터(X_train)를 사용해 40~41라인에서 설정한 전처리 과정과 모델을 학습한다.
pipeline.fit(X_train, y_train)

# TODO B07-2: pipeline.predict(X_valid)로 검증 입력을 예측하세요. 현재 값은 임시값입니다.
#46라인에서 학습이 끝난 모델에게 데이터(X_valid)를 넘겨주어 예측을 진행한다.
#predict은 Pipeline객체에서 사용할 수 있는 메서드로 입력 데이터를 바탕으로 예측을 진행하는 역할을 한다.
prediction = pipeline.predict(X_valid)

#결과를 정리한다.
result = valid[["request_id", "failed"]].copy()#검증용 데이터의 요청id와 실패 여부만 선택하여 복사후 result에 저장
result["predicted"] = prediction # predicted열 생성후 51라인에서 생성한 predicted 데이터를 해당 열에 삽입
#54,55 라인에서 만든 데이터의 결과 표를 valid_predictions.csv파일로 생성한다.
result.to_csv(HERE / "valid_predictions.csv", index=False, encoding="utf-8-sig")
#실제 정답과 모델의 예측값이 일치하는 비율을 계산한다.
accuracy = (result["failed"] == result["predicted"]).mean()

#결과 출력
print("분할:", len(train), len(valid), len(test)) # 학습용, 검증용 ,시험용 데이터의 행수 출력
print(result.to_string(index=False)) # 결과표 전체를 행번호(인덱스)없이 문자열 형태로 출력
print("valid 정확도:", round(accuracy, 2)) # 59라인에서 계산한 valid의 정확도를 소수점 둘째 자리까지 출력
print("저장: valid_predictions.csv / test는 사용하지 않았습니다.")
