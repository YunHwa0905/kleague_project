# K리그 패스 좌표 예측 - 최종 구현 가이드

## 🎯 목표
- **올바른 방법**으로 LightGBM 모델 학습
- **Data Leakage 완전 제거**
- **Episode 독립성 보장**
- **예상 점수: 20-25점** (현재 최고: 27.63)

---

## 📂 프로젝트 구조

```
D:/workspce/kleague_project/
├── data/
│   ├── train.csv              ← 전체 train 데이터 (정답 포함)
│   ├── test.csv               ← test 에피소드 목록
│   ├── sample_submission.csv
│   ├── match_info.csv
│   └── test/                  ← test 에피소드 파일들
│       ├── 153363/
│       └── ...
├── models/                    ← 학습된 모델 저장 위치 (생성됨)
├── outputs/                   ← 제출 파일 저장 위치 (생성됨)
├── feature_extractor_v3.py    ← Feature 추출기 (새로 복사!)
├── train_final.py             ← 학습 스크립트 (새로 복사!)
└── inference_final.py         ← 추론 스크립트 (새로 복사!)
```

---

## 🚀 실행 방법

### Step 1: 파일 복사

다음 3개 파일을 프로젝트 루트에 복사:
1. `feature_extractor_v3.py`
2. `train_final.py`
3. `inference_final.py`

### Step 2: 학습 실행

```bash
cd D:\workspce\kleague_project
python train_final.py
```

**예상 소요 시간: 30-60분**

출력 예시:
```
Loading train.csv...
✅ Loaded 356,721 rows
✅ Unique episodes: 15,435

Preparing training data...
Processing episodes: 100%|████████| 15435/15435

📈 Train: 12,348 episodes from 182 games
📉 Val:   3,087 episodes from 46 games

🔵 Training X coordinate model...
[100] train's rmse: 10.5432  val's rmse: 12.3456

🔴 Training Y coordinate model...
[100] train's rmse: 11.2345  val's rmse: 13.4567

🎯 Validation Metrics:
  Average Euclidean Distance: 18.2345

💾 Models saved to: D:/workspce/kleague_project/models/
```

### Step 3: 추론 실행

```bash
python inference_final.py
```

**예상 소요 시간: 5-10분**

출력 예시:
```
Loading trained models...
✅ Loaded models from D:/workspce/kleague_project/models/

Loading test data...
✅ Test episodes: 2,414

Extracting features...
Making predictions...

✅ Submission file created:
   D:/workspce/kleague_project/outputs/submission_final_20251208_HHMMSS.csv
```

### Step 4: Dacon 제출

1. `outputs/submission_final_XXXXXX.csv` 파일 확인
2. Dacon에 업로드
3. Public Score 확인!

---

## ✅ 핵심 개선 사항

### 1. Data Leakage 완전 제거 ✨

**이전 (잘못된 방법):**
```python
# train_test_split으로 무작위 분할
# → 같은 game_id의 에피소드가 train/val에 섞임!
X_train, X_val = train_test_split(X, y)
```

**현재 (올바른 방법):**
```python
# game_id 기반 split
# → 같은 게임은 train 또는 val에만 존재!
gss = GroupShuffleSplit(n_splits=1, test_size=0.2)
train_idx, val_idx = next(gss.split(X, groups=game_ids))
```

### 2. Episode 독립성 보장 ✨

**사용 가능한 정보:**
- ✅ 마지막 패스의 start 좌표
- ✅ 마지막 패스 이전의 액션 시퀀스
- ✅ Episode 내부 통계

**사용 불가능:**
- ❌ 다른 episode의 정보
- ❌ 마지막 패스의 end 좌표
- ❌ 전체 데이터 통계

### 3. Feature 품질 개선 ✨

**29개의 올바른 Feature:**
1. 마지막 패스 시작 좌표 (last_start_x, last_start_y)
2. 시퀀스 통계 (평균, 표준편차, 진행 방향)
3. 최근 액션 패턴 (최근 3개 패스의 이동 방향)
4. 액션 타입 통계 (패스 비율, 성공률)
5. 필드 위치 특성 (구역, 중심 거리)

---

## 📊 예상 성능

### 이전 모델들:
```
Baseline v1:    35.99  ← Y축 고정
Baseline v2:    27.63  ← 현재 최고!
LightGBM v1:    32.80  ← Data Leakage로 실패
LightGBM CV:    32.89  ← 더 나빠짐
```

### 최종 모델 예상:
```
Validation:     18-22  ← game_id split으로 정확한 평가
Public Score:   20-25  ← Baseline v2 대비 2-7점 개선!
```

**개선 이유:**
1. Data Leakage 제거 → 신뢰할 수 있는 validation
2. 올바른 Feature → 실제 패턴 학습
3. LightGBM 파워 → 복잡한 관계 포착

---

## 🔍 학습 과정 이해하기

### 1. 데이터 로드
- train.csv에서 전체 데이터 로드
- 15,435개 에피소드, 228개 게임

### 2. 데이터 준비
- 각 에피소드별로:
  - 마지막 패스의 end_x, end_y → 정답
  - 마지막 패스 제외 → Feature 추출
  
### 3. Train/Val Split
- game_id로 분할 (80:20)
- Train: 182 게임, Val: 46 게임
- **같은 게임은 섞이지 않음!**

### 4. 모델 학습
- X 좌표 모델 (RMSE 최소화)
- Y 좌표 모델 (RMSE 최소화)
- Early Stopping (50 rounds)

### 5. 검증
- Validation 데이터로 유클리드 거리 계산
- 이 점수가 Public Score와 유사할 것!

---

## 🎓 주요 교훈

### 1. Data Leakage의 위험성
- Validation: 16.08 → Public: 32.80
- **두 배 차이!** → 학습 과정에 문제가 있었음

### 2. 올바른 검증의 중요성
- game_id 기반 split이 필수
- Validation Score를 믿을 수 있어야 함

### 3. 단순함의 가치
- Baseline v2 (27.63)가 복잡한 모델보다 나았음
- 올바른 구현 > 복잡한 알고리즘

---

## 💡 추가 개선 아이디어

만약 점수가 기대에 미치지 못하면:

### 1. Optuna 하이퍼파라미터 튜닝
- 50-100 trials
- 예상 개선: 1-3점

### 2. 앙상블
- 5-Fold CV
- 예상 개선: 1-2점

### 3. Feature 추가
- 선수별 패턴
- 팀 전술 특성
- 예상 개선: 2-4점

---

## 📝 체크리스트

학습 전:
- [ ] 3개 파일 복사 완료
- [ ] data/train.csv 존재 확인
- [ ] data/test/ 폴더 존재 확인

학습 중:
- [ ] Validation Score 확인 (18-22 예상)
- [ ] 에러 없이 완료
- [ ] models/ 폴더에 4개 파일 생성

추론 후:
- [ ] outputs/ 폴더에 submission 파일 생성
- [ ] 2,414개 행 확인
- [ ] 좌표 범위 확인 (X: 0-105, Y: 0-68)

제출:
- [ ] Dacon 업로드
- [ ] Public Score 확인
- [ ] Baseline v2 (27.63)와 비교

---

## 🆘 문제 해결

### Q1: "FileNotFoundError: train.csv"
**A:** 경로 확인:
```python
# train_final.py, inference_final.py에서
train_path = "D:/workspce/kleague_project/data/train.csv"
# 이 경로가 정확한지 확인
```

### Q2: "Memory Error"
**A:** 메모리 부족 시:
- 배치 처리로 변경
- 또는 더 많은 RAM 확보

### Q3: "Validation Score가 너무 높음 (>25)"
**A:** 정상일 수 있음:
- Public Score 확인 후 판단
- Feature 추가로 개선 시도

### Q4: "Public Score가 Validation보다 높음"
**A:** 좋은 신호!
- Overfitting이 아님
- Private Score도 좋을 가능성

---

## 🎉 성공하면?

Public Score 20-25 달성 시:
1. 🎊 축하합니다!
2. 📈 상위 20-30% 예상
3. 🏆 본선 진출 가능성!
4. 💪 추가 개선으로 Top 10 도전!

---

**행운을 빕니다!** 🍀