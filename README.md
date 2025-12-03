# 🧠 EEG

# EEGNet 기반 단일 인코더의 Release 간 제로-샷 EEG 반응시간 예측

> **Healthy Brain Network (HBN) – Contrast Change Detection (CCD) EEG만을 이용한  
> EEGNet 기반 제로-샷(Zero-Shot) 반응시간 회귀 모델**

---

## 프로젝트 개요 (Overview)

이 프로젝트는 **HBN-EEG의 CCD 과제**에서 얻은 **128채널 EEG**만을 사용해,  
**EEGNet 기반 단일 인코더 + 회귀 헤드**로 **버튼 반응시간(Response Time, RT)** 을 예측하는 모델을 구현하고 평가하는 것을 목표로 합니다.  

- 입력: trial 단위 **EEG 신호 (X)** 및 피험자 정보 (P)  
- 출력: trial 단위 **반응시간 (RT)**  
- 설정: **학습에 포함되지 않은 Release / 피험자에 대한 제로-샷 회귀 성능 평가** 

---

## Motivation

EEG(Electroencephalography)는 두피 전극으로 뇌의 전기 활동을 **비침습적·실시간** 측정할 수 있는 강력한 도구지만, 다음과 같은 이유로 분석이 어렵습니다. 

- **낮은 SNR (Signal-to-Noise Ratio)**  
- 안구/근전도/환경 잡음 및 두피 혼합으로 인한 **신호 해석 및 공간 분리의 어려움**
- 피험자·세션·장비 변화에 따른 **도메인 갭(domain gap)**  
  → 학습 시점과 테스트 시점의 분포 차이로 **성능 괴리 발생**

기존 **EEG 전이학습(Transfer Learning)** 접근은 도메인 갭을 줄이기 위해 여전히 **보정(calibration) 데이터**를 필요로 하는 경우가 많고,  
**cross-task / cross-subject 일반화**를 완전히 해결하지 못하고 있습니다. 
---

## Related Work: EEG Foundation Challenge

본 프로젝트는 **EEG Foundation Challenge: From Cross-Task to Cross-Subject EEG Decoding** 벤치마크 문제 설정을 기반으로 합니다. 

- **HBN-EEG**:  
  - 다양한 인지 과제(총 6개 이상) + 다수 피험자가 섞여 있는 **대규모 EEG 데이터셋**
  - 과제별로 모델을 따로 만드는 대신, **단일 공통 EEG 표현(single shared representation)** 을 학습하는 문제를 제시
- **Challenge 1 – Cross-Task Transfer (Zero-Shot Cross-Domain Generalization)**  
  - 입력: trial 단위 EEG 신호  \(X\) + 피험자 정보 \(P\)  
  - 출력: RT, Accuracy 등 행동 지표 \(Y\)  
  - 새로운 과제/새 피험자에 대해 **추가 파인튜닝 없이** 성능을 평가하는 제로-샷 설정

본 프로젝트는 이 중 **CCD task만을 사용하는 제로-샷 RT 회귀 세팅**에 초점을 맞춥니다.

---

## 연구 목표 (Goal)

> **CCD 과제 EEG만을 사용한, EEGNet 기반 제로-샷 반응시간 회귀 모델 구축**


1. **EEGNet 인코더 + MLP 회귀 헤드**를 end-to-end로 학습
2. **Non-CCD task로는 사전학습(pretraining)을 수행하지 않고**,  
   CCD task 데이터만으로 학습
3. 학습에 포함되지 않은 Release / 피험자에 대해
   - **추가 파인튜닝 없이** RT를 예측하는 제로-샷 설정에서
   - EEG만으로 달성 가능한 **반응시간 예측 성능**을 정량적으로 평가

---

## 데이터셋: HBN-EEG (Healthy Brain Network EEG)

- 제공 기관: **Child Mind Institute**  
- 대상: **5–21세 아동·청소년**  
- 채널 수: **128채널 EEG**  
- 포맷: **BIDS 형식**  
- 각 세션에 대해  
  - 자극(onset) 이벤트  
  - 반응(response) 이벤트  
  주석(annotation) 제공
- 이 프로젝트에서는 **12개 Release 중 1–5 Release만 활용**  

### 과제(Task) 구성


- **Active-task (능동 과제)**  
  - Surround Suppression (**SuS**)  
  - Sequence Learning (**SL**)  
  - Contrast Change Detection (**CCD**)  
  - Symbol Search (**SS**)  

- **Passive-task (수동 과제)**  
  - Movie Watching (**MW**)  
  - Resting State (**RS**)  

이 중 본 프로젝트는 **CCD task**만을 사용하여 모델을 학습·평가합니다.

### CCD vs Non-CCD 상관관계

 **CCD task와 Non-CCD task 간 RT 상관관계**를 계산한 결과,  
Non-CCD 데이터는 CCD RT 예측에 **큰 기여를 하지 못할 것**으로 나타났습니다.  

이에 따라,  
- **Non-CCD 데이터로 사전학습을 수행하지 않고**,  
- **CCD task 데이터만으로 모델을 학습**하는 설계를 채택했습니다. 


---

## 전처리 파이프라인 (Preprocessing)
 

1. **필터링 및 재표본화 (Filtering & Resampling)**  
   - EEG 신호 대역 필터링  
   - 일정 샘플링 주파수로 resample  
   - 자세한 필터 파라미터는 구현 코드 기준

2. **공통 채널 선택 (Common Channel Selection)**  
   - 모든 세션/피험자에서 공통적으로 사용 가능한 채널만 선택

3. **유효 트라이얼 선택 (Valid Trial Selection)**  
   - 자극/반응 이벤트 주석 기반으로  
   - 행동 반응(버튼 press)이 존재하고,  
   - 전처리 기준을 만족하는 trial만 유지

4. **반응 기준 구간 추출 (Response-Locked Epoching)**  
   - 버튼 반응 시점을 기준으로 일정 시간 구간(epoch)을 추출  
   - 최종 입력 형태: \((N, C, T))

5. **채널 표준화 – 피험자별 Normalization**  
   - 피험자마다 채널 분포가 다른 문제를 줄이기 위해  
   - 동일 피험자 내에서 채널 기준 표준화 수행  

6. **RT(반응시간) 표준화 – 피험자별 Normalization**  
   - 피험자마다 전반적인 RT 수준이 다르므로  
   - 피험자별로 RT 분포를 정규화하여,  
     모델이 **상대적인 trial 간 변동성**에 더 집중하도록 설정  


---

## 모델 구조 (Model Architecture)

본 프로젝트의 모델은 **EEGNet 기반 단일 인코더 + 회귀 헤드(MLP)** 구조로 구성됩니다.

![모델 구조](figures/EEG%20모델%20구조.png)

- **Backbone**:  
  - EEGNet 인코더  
  - 128채널 EEG를 입력으로 받아  
  - trial-level latent representation을 출력

- **Head (Regression Head)**:  
  - EEGNet 인코더 출력 feature를 **Flatten**  
  - 1개 이상의 **Fully Connected (Linear) layer**로 구성된 MLP  
  - 최종 출력 차원: **스칼라 RT** (표준화된 반응시간)

- **학습 방식**:  
  - 인코더(EEGNet)와 회귀 헤드를 **end-to-end로 공동 학습**

---

## 평가 지표 (Evaluation Metrics)

본 프로젝트에서 사용한 평가지표는 다음과 같습니다.   

1. **MAE (Mean Absolute Error)**  
   - 예측 RT와 실제 RT 간의 **절대 오차의 평균**

2. **RMSE (Root Mean Squared Error)**  
   - 예측 오차 제곱 평균의 제곱근  
   - 큰 오차에 더 큰 패널티를 부여

3. **nRMSE (normalized RMSE)**  
   - RMSE를 특정 기준(예: 값 범위, 표준편차 등)으로 정규화한 값  

---

## 실험 설정 (Experimental Settings)

### Release 조합 – 제로-샷 설정

- HBN-EEG의 **Release 1–5**를 사용하여,  
  여러 가지 **Train/Test Release 조합**(총 5가지)을 구성했습니다. 

- 각 실험에서  
  - 일부 Release는 **학습(train) 전용**,  
  - 다른 Release는 **테스트(test) 전용**으로 사용  
  - 테스트 Release에 포함된 피험자는 학습에 포함되지 않은 **새 피험자**

- 테스트 시에는 **학습된 가중치를 그대로 사용**하며,  
  **task-specific / subject-specific 파인튜닝은 전혀 수행하지 않는  
  완전 제로-샷(Zero-Shot) 설정**입니다.

---

## 실험 결과 요약 (Experimental Results)


- **5가지 Train/Test Release 조합**에서의 제로-샷 Test 성능:

| 지표 | 범위 (Test 기준, CCD Task) |
|------|----------------------------|
| **MAE** | 약 **0.2266 – 0.2376 s** |
| **RMSE** | 약 **0.2919 – 0.3156 s** |
| **nRMSE** | 약 **0.83 – 0.98** |
| **상관계수 r** | 약 **0.27 – 0.57** |

- Release별 차이:
  - **R3 / R5**를 테스트로 둔 경우  
    → 상관계수 \( r \approx 0.52 – 0.57 \)로 **가장 높은 성능**
  - **R1**을 테스트로 둔 경우  
    → 상관계수 \( r \approx 0.27 \)로 **가장 낮은 성능**

### 해석


- **CCD EEG만으로도**  
  trial 간 **반응시간 변동성을 일정 수준까지 예측** 가능
- 다만, **Release별 데이터 분포 및 품질 차이**가  
  제로-샷 성능에 **상당한 영향을 미침**  
  → 동일한 모델이라도 Test Release에 따라 상관계수가 크게 달라짐


---

## 결론 (Conclusion)


1. **단일 EEGNet 인코더 + MLP 헤드**만으로도  
   - 학습에 포함되지 않은 Release 및 피험자에 대해  
   - 버튼 반응시간을 **수백 ms 이내의 오차**로 안정적으로 예측 가능

2. **5가지 Train/Test 조합 모두에서**,  
   - 새로운 데이터에 대해 trial 간 RT 변동을 **눈에 보이는 수준으로 따라가는**  
     제로-샷 회귀 성능을 확인  
   - 일부 설정에서는 **상당히 높은 상관(r > 0.5)** 도 달성

3. 즉,  
   - 별도의 사전학습(pretraining)이나 복잡한 구조 없이도,  
   - **CCD EEG만으로 cross-subject 제로-샷 반응시간 예측이  
     실용적인 수준까지 가능함**을 보여줌

---

## 향후 과제 (Future Work)


1. **데이터 증강 (Data Augmentation)**  
   - 다양한 노이즈, 시간 왜곡, 채널 드랍아웃 등 EEG 특화 증강 기법 도입 가능

2. **더 다양한 피험자 대상 학습 및 테스트**  
   - 연령, 임상 특성, 세션 수 등 다양성을 확장하여  
     제로-샷 일반화 범위 확인

3. **정규화 / 모델 Ablation Study**  
   - robust z-score vs 일반 z-score  
   - subject-wise vs global 정규화  
   - 이러한 설정에 따른 제로-샷 성능 변화 비교

4. **Few-Shot 시나리오 평가**  
   - Test Release에서 **소량의 라벨만 사용해 헤드만 미세조정**  
   - 완전 제로-샷과의 성능 차이를 정량적으로 비교

---
