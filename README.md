# MLflow Study

Iris 분류 예제로 MLflow의 실험 기록과 조회 방법을 공부한 노트북입니다. PyTorch 모델을 학습하면서 파라미터·학습 정확도·모델을 남기고, 기록된 모델을 지표 순서로 조회합니다.

## 들어 있는 내용

| 노트북 | 코드와 저장된 출력에서 확인한 내용 |
| --- | --- |
| [Deep_learning_quick_start.ipynb](Deep_learning_quick_start.ipynb) | Iris 분류기 학습, 10 epoch 간격의 모델·지표 기록, 학습 정확도로 정렬한 모델 조회 |
| [using_the_mlflow_client_API.ipynb](using_the_mlflow_client_API.ipynb) | `MlflowClient`로 실험 목록과 Default 실험의 이름·상태 조회 |

첫 노트북은 train/test 데이터를 나누지만, 정확도 계산과 모델 비교에는 train 데이터만 사용합니다. 저장된 출력은 당시 실행 기록이며, 이번 문서 정리에서 학습을 다시 실행한 결과는 아닙니다.

모델 배포, Model Registry 등록·승격, 여러 실험의 자동 비교는 구현 범위에 포함되지 않습니다.

## 읽는 순서

1. `IrisClassifier`와 데이터 준비 셀
2. `mlflow.start_run()` 안의 학습·기록 루프
3. `mlflow.search_logged_models()`를 이용한 조회
4. Client API 노트북의 실험 목록 조회

## 실행 준비

모델 기록·검색 코드는 MLflow 3 계열 API를 사용합니다. 의존성 버전이 고정돼 있지 않아 현재 환경에서의 전체 재실행은 별도 확인이 필요합니다.

```bash
python -m pip install "mlflow>=3,<4" torch pandas scikit-learn jupyterlab
python -m jupyterlab
```

Client 노트북은 `http://127.0.0.1:5001`에 연결합니다. 해당 주소에 서버가 없으면 별도 터미널에서 실행하거나 `tracking_uri`를 바꿉니다.

```bash
mlflow server --host 127.0.0.1 --port 5001
```

학습 노트북에는 tracking URI 지정이 없습니다. 두 노트북을 같은 서버에 연결하려면 학습 전에 `mlflow.set_tracking_uri("http://127.0.0.1:5001")`를 지정해야 합니다.