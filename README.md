# MLflow Study

MLflow로 실험과 모델을 기록하고 조회하는 방법을 익힌 노트북 모음입니다. Iris 분류 예제로 학습 중 지표와 모델을 남기고, Client API로 실험 정보를 조회합니다.

## 노트북

| 파일 | 내용 |
| --- | --- |
| [Deep_learning_quick_start.ipynb](Deep_learning_quick_start.ipynb) | PyTorch Iris 분류기 학습, 파라미터·정확도·모델 기록, 기록한 모델의 지표 비교 |
| [using_the_mlflow_client_API.ipynb](using_the_mlflow_client_API.ipynb) | `MlflowClient` 연결과 실험 목록 조회 |

분류 예제는 10 epoch마다 모델과 학습 데이터의 정확도를 기록합니다. 코드는 train/test로 데이터를 나누지만, 현재 모델 순위에는 학습 정확도를 사용합니다. 이 값을 독립 test 성능으로 보지는 않습니다.

## 실행 준비

Python 가상환경에 필요한 패키지를 설치하고 JupyterLab에서 노트북을 엽니다. 모델 기록과 검색에는 MLflow 3 계열 API를 사용합니다.

```bash
python -m pip install "mlflow>=3,<4" torch pandas scikit-learn jupyterlab
python -m jupyterlab
```

Client 예제는 `http://127.0.0.1:5001`의 tracking server에 연결합니다. 별도 터미널에서 아래처럼 서버를 실행하거나, 노트북의 `tracking_uri`를 사용 중인 주소로 바꿉니다.

```bash
mlflow server --host 127.0.0.1 --port 5001
```

학습 예제에는 tracking URI 지정이 없으므로 MLflow의 기본 설정을 사용합니다. 두 노트북이 같은 서버를 사용하게 하려면 학습 전에 `mlflow.set_tracking_uri("http://127.0.0.1:5001")`를 지정합니다.

## 다음에 확인할 것

의존성 버전을 고정한 실행 환경은 아직 없습니다. 다음에는 test 데이터 평가와 실행별 설정 비교를 추가해, 기록한 모델을 어떤 기준으로 선택할지까지 정리하려고 합니다.
