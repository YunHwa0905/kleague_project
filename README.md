# K리그 패스 예측

K리그 경기 데이터를 활용해 에피소드의 마지막 패스 종착점(end_x, end_y)을 예측하는 머신러닝 대회 참가 저장소입니다.

---

## 기술 스택

![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=flat&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02A05C?style=flat&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EC5F2A?style=flat&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=flat&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-4A90E2?style=flat&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-11557C?style=flat&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

## 프로젝트 구조

```
kleague_project/
├── src/                              # 핵심 모듈
│   ├── config.py                     # 경로·상수·하이퍼파라미터 중앙 관리
│   ├── data_loader.py                # 에피소드 데이터 로딩
│   ├── features.py                   # FeatureExtractor 기본 클래스
│   └── utils.py                      # 공통 유틸리티 (거리·각도 계산 등)
│
├── data/                             # 대회 데이터셋
│   ├── train.csv                     # 학습 데이터 (에피소드 액션 시퀀스)
│   ├── test.csv                      # 예측 대상 에피소드
│   ├── match_info.csv                # 경기 메타 정보
│   ├── sample_submission.csv         # 제출 양식
│   └── test/                         # 테스트 에피소드 개별 파일
│
├── models/                           # 학습된 모델 파일
│   ├── lgb_v4_x_fold{1-5}.txt        # LightGBM 5-Fold 모델
│   ├── xgb_v4_x_fold{1-5}.json       # XGBoost 5-Fold 모델
│   └── cat_v4_x_fold{1-5}.cbm        # CatBoost 5-Fold 모델
│
├── outputs/                          # 제출 CSV
├── notebooks/
│   └── 01_eda.py                     # 데이터 탐색 (EDA)
│
├── feature_extractor_v3.py           # Feature v3 (67개)
├── feature_extractor_v4.py           # Feature v4 (67개 개선)
├── feature_extractor_v4_enhanced.py  # Feature v4 Enhanced (85개)
│
├── phase1_1_lgb_v4.py                # Phase 1 학습: LightGBM
├── phase1_2_xgb_v4.py                # Phase 1 학습: XGBoost
├── phase1_3_cat_v4.py                # Phase 1 학습: CatBoost
├── phase2_stacking.py                # Phase 2: Ridge 스태킹 앙상블
│
├── train.py                          # LightGBM 기본 학습
├── train_cv.py / train_5fold.py      # K-Fold CV 학습
├── train_optuna.py                   # Optuna 하이퍼파라미터 튜닝
│
├── Inference_final.py                # 최종 추론 (제출용)
├── inference_weighted_final.py       # 가중 앙상블 추론
│
└── requirements.txt
```

---

## 실행 방법

```bash
# 1. 의존성 설치
pip install -r requirements.txt
pip install xgboost catboost optuna

# 2. EDA 실행
python notebooks/01_eda.py

# 3. Phase 1 — 개별 모델 학습 (5-Fold × 3 알고리즘)
python phase1_1_lgb_v4.py
python phase1_2_xgb_v4.py
python phase1_3_cat_v4.py

# 4. Phase 2 — 스태킹 앙상블
python phase2_stacking.py

# 5. 최종 추론 및 제출 파일 생성
python Inference_final.py
```

> `data/` 폴더에 `train.csv`, `test.csv`, `sample_submission.csv`가 있어야 합니다.  
> 모델 저장 경로(`models/`)는 `src/config.py`에서 수정하세요.

---

## 학습 내용

| 파일 / 모듈 | 주요 학습 내용 |
|---|---|
| `notebooks/01_eda.py` | 에피소드 분포 탐색, 필드 좌표 시각화, 액션 타입 분석 |
| `src/features.py` | 시퀀스 통계·위치·방향·시간·팀 정보 Feature 설계 |
| `feature_extractor_v4_enhanced.py` | 67 → 85개 Feature 확장, 골문 거리·각도·압박 지수 추가 |
| `phase1_1_lgb_v4.py` | LightGBM 5-Fold CV 학습, 조기 종료, Feature Importance |
| `phase1_2_xgb_v4.py` | XGBoost 5-Fold CV 학습, tree_method GPU 설정 |
| `phase1_3_cat_v4.py` | CatBoost 5-Fold CV 학습, 범주형 변수 처리 |
| `phase2_stacking.py` | 15개 모델(5 Fold × 3 알고리즘) → Ridge 메타 모델 스태킹 |
| `train_optuna.py` | Optuna TPE 샘플러로 하이퍼파라미터 자동 탐색 |
| `Inference_final.py` | 학습 모델 로드 → 에피소드별 Feature 추출 → 좌표 예측 |

---

## 개발 환경

- **Python 3.11** 이상
- **패키지 관리**: pip / venv

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
pip install xgboost catboost optuna
```

- **IDE** — VS Code / PyCharm
- **데이터 탐색** — Jupyter Notebook 또는 `.py` 스크립트 직접 실행
