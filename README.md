# MLflow Study

Iris 분류 예제로 MLflow의 실험 기록과 조회 방법을 공부한 노트북입니다. train 데이터로 PyTorch 모델을 학습하고, 분리한 test 데이터의 정확도를 모델별로 기록해 비교합니다. 학습 정확도는 별도 보조 지표로 남깁니다.

## 들어 있는 내용

| 노트북 | 내용 |
| --- | --- |
| [Deep_learning_quick_start.ipynb](Deep_learning_quick_start.ipynb) | Iris 분류기 학습, 10 epoch 간격의 train/test 정확도·모델 기록, test 정확도로 정렬한 모델 조회 |
| [using_the_mlflow_client_API.ipynb](using_the_mlflow_client_API.ipynb) | `MlflowClient`로 실험 목록과 Default 실험의 이름·상태 조회 |

고정 seed 42와 클래스별 비율을 유지하는 분할로 train 120개/test 30개를 나눕니다. 손실 계산·역전파에는 train만 사용합니다. 평가는 `eval()`과 `no_grad()`에서 실행한 뒤 원래 모델 모드로 복원합니다.

MLflow에는 `train_accuracy`와 `test_accuracy`를 각 데이터셋에 연결해 기록하고 `test_accuracy` 내림차순으로 조회합니다. test 지표로 체크포인트를 비교하는 예제이므로, 모델 선택에 쓰지 않은 최종 독립 test 성능은 아닙니다. 그런 평가에는 별도의 validation/test 분할이 필요합니다.

모델 배포, Model Registry 등록·승격, 여러 실험의 자동 비교는 구현 범위에 포함되지 않습니다.

## 읽는 순서

1. `IrisClassifier`와 데이터 준비 셀
2. `mlflow.start_run()` 안의 학습·기록 루프
3. `mlflow.search_logged_models()`를 이용한 조회
4. Client API 노트북의 실험 목록 조회

## 실행 준비

모델 기록·검색 코드는 MLflow 3.17 계열에서 확인한 API를 사용합니다.

```bash
python -m pip install "mlflow>=3.17,<4" torch pandas scikit-learn jupyterlab
python -m jupyterlab
```

Client 노트북은 `http://127.0.0.1:5001`에 연결합니다. 해당 주소에 서버가 없으면 별도 터미널에서 실행하거나 `tracking_uri`를 바꿉니다.

```bash
mlflow server --host 127.0.0.1 --port 5001
```

학습 노트북에는 tracking URI 지정이 없습니다. 두 노트북을 같은 서버에 연결하려면 학습 전에 `mlflow.set_tracking_uri("http://127.0.0.1:5001")`를 지정해야 합니다.

## 테스트

```bash
python -m unittest discover -s tests -v
```

[테스트](tests/test_test_evaluation.py)는 노트북의 코드 셀을 Python 모듈로 읽어 실제 CPU 학습 101 step과 MLflow 로컬 SQLite 기록을 실행합니다. 외부 tracking server는 사용하지 않습니다.

- train/test 행이 겹치지 않고 손실 계산에 train 정답만 들어가는지 확인
- 체크포인트 11개의 train/test 지표와 데이터셋 연결 확인
- 저장한 모델 11개를 다시 불러와 test 정확도를 독립적으로 계산하고 기록값·정렬 순서와 비교
- 평가 중 gradient 비활성화와 모델 모드 복원 확인

과거 학습 정확도 출력은 혼동을 피하기 위해 비웠습니다. 테스트 산출물과 가상환경은 Git에 포함하지 않습니다.

2026-10-24에 Python 3.11.16, PyTorch 2.7.1, MLflow 3.17.2, scikit-learn 1.9.1의 CPU 환경에서 위 테스트 4개를 통과했습니다. 실제 101회 학습 갱신과 체크포인트 11개의 저장·재로딩을 확인한 범위입니다.
