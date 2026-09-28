# 평점 희소성 환경에서 정규화 행렬분해의 전역 스케일링과 차원별 스케일링 비교

논문 **"Global versus Dimension-Specific Scaling in Normalized Matrix Factorization under Rating Sparsity"**의 재현성 저장소입니다.

## 버전

**v1.4.0 — 투고용 공개 버전**

## 포함된 분석

- 주요 벤치마크: MovieLens 100K, MovieLens 1M
- 외부 재현: Book-Crossing, FilmTrust, Jester
- 사용자별 약 80/10/10 train/validation/test 분할
- validation/test 고정, training set만 중첩 희소화
- 명목상 유지율: 100%, 75%, 50%, 25%, 10%
- 사용자별 최소 5개 training rating 유지
- 모형: BiasOnly, BiasMF, CosineMF, NormalizedMF
- 지표: RMSE, MAE
- 잠재차원 및 weight decay 민감도 분석
- split seed 강건성 분석
- 10% 조건 warm-item 분석
- MovieLens 사용자 군집 쌍체 부트스트랩
- Jester 원척도 민감도 분석

초기화 seed 반복은 모집단 표본이 아니라 **최적화 변동성**을 확인하기 위한 것입니다. 사용자 군집 부트스트랩은 학습된 모형을 고정한 상태에서 test 사용자 구성 변화에 따른 변동을 평가합니다.

## 데이터

원자료는 저작권 및 이용조건 때문에 저장소에 포함하지 않습니다. 필요한 파일과 전처리 방법은 [`data/README.md`](data/README.md)를 참고하십시오.

## 권장 실행 순서

```bash
python scripts/01_main_experiments.py --dataset both
python scripts/02_sensitivity_analysis.py
python scripts/03_warm_item_robustness.py
python scripts/04_split_robustness.py
python scripts/07_external_validation.py --dataset all
python scripts/07_external_validation.py --dataset Jester --jester-native-scale
python scripts/05_make_summaries.py
python scripts/08_user_cluster_bootstrap.py --dataset both --bootstrap-reps 10000
python scripts/06_make_final_figures.py
```

분석량이 많아 MovieLens 1M, Jester, bootstrap은 실행 시간이 길 수 있습니다.

## 주요 모형

- **BiasOnly**: 전체 평균 + 사용자 편향 + 아이템 편향
- **BiasMF**: 편향항 + 일반적인 잠재요인 내적
- **CosineMF**: 사용자/아이템 벡터 L2 정규화 + 하나의 양의 전역 scale
- **NormalizedMF**: 사용자/아이템 벡터 L2 정규화 + 잠재차원별 양의 scale

논문의 핵심 직접 비교는 CosineMF와 NormalizedMF입니다.

## 주요 고정 설정

- 잠재차원: K = 32
- Adam learning rate = 0.015
- weight decay = 1e-5
- 사용자별 최소 training rating = 5
- 기본 split seed = 20260918
- MovieLens 100K initialization seeds = 10개
- MovieLens 1M = 5개
- Book-Crossing / FilmTrust = 5개
- Jester = 3개

## 결과 파일

`results/`에는 논문 및 보충자료에 사용한 주요 summary CSV를 포함합니다. 원자료에서 재생성 가능한 대용량 long-format 결과는 Git 저장소에 유지하지 않아도 됩니다.

사용자 군집 쌍체 bootstrap 결과는 `results/user_cluster_bootstrap_ci.csv`에 있습니다.

## 재현성 주의사항

동일한 seed를 사용하더라도 운영체제, CPU/GPU, BLAS backend, PyTorch build에 따라 마지막 소수점 수준의 차이가 발생할 수 있습니다. 재현 시에는 마지막 자리의 완전 일치보다 논문에 보고한 평균값과 CosineMF-NormalizedMF의 쌍체 차이 방향 및 크기를 확인하는 것이 적절합니다.

## 라이선스

이 저장소의 자체 작성 코드는 MIT License로 배포합니다. MovieLens, Book-Crossing, FilmTrust, Jester 등의 제3자 데이터는 포함하지 않으며 각 데이터셋의 별도 이용조건을 따릅니다.
