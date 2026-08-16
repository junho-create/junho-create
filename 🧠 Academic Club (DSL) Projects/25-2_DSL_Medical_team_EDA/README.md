## 프로젝트 제목 
CGM: 연속 혈당 모니터링 데이터를 활용한 혈당 변화 패턴 분석 및 INSULIN SENSITIVITY 정량화


## 프로젝트 멤버
- 서승범(팀장) 박시현, 양재화, 신태희, 여준호
## 프로젝트 기간
* 2025.07.08. ~ 2025.08.05.
## 프로젝트 목표
> 1. 인슐린 투입과 탄수화물 섭취, 심박수에 따른 혈당 변화 패턴을 파악한다
> 2. 개인별 인슐린 주입에 따른 혈당 변화량 차이의 정도 (Insulin sensitivity)를 계산하고, Insulin sensitivity의 차이에 영향을 미치는 요인을 파악한다
> 3. 탄수화물 섭취와 심박수 증가, 그로 인한 인슐린의 강하 효율을 파악한다
## 사용 데이터
> * PEDAP: PEDAP Trial (2023, NEJM)에 사용된 102명의 1형 당뇨 소아 환자 데이터, Blood glucose reading, Carbohydrate intake, Insulin dosage, Personal information 포함
> * BrisT1D:  1형 당뇨를 앓는 24명의 young adult에게서 수집된 데이터, Blood glucose readings, Carbohydrate intake, Smartwatch activity, Insulin dosage, Heart rate 포함
> * HUPA-UCM: 1형 당뇨를 앓는 25명의 사람들에게서 수집된 데이터, Blood glucose readings, Steps (걸음수), Insulin dasage, Calories burned, Carbohydrate grams, Heart rate 포함
## 데이터 분석 방법
> * Linear mixed model (LMM)을 이용한 개인별 Insulin sensitity 추정
혈당 변화를 종속 변수, Insulin, 탄수화물 섭취를 독립 변수로 하고 이들의 환자별 random effect를 고려한 linear mixture model를 구성해 환자의 insulin sensitivity, 탄수화물 섭취와 insulin 투여가 혈당 변화에 영향을 미치는 시간 간격 파악
> * 시계열 모형 (ARIMA)를 이용한 insulin sensitivity 추정
외생 변수 (insulin, 탄수화물 섭취), 24시간 주기 (fourier 고조파)를 포함한 환자별 ARIMA 모델로 개인별 인슐린 민감도를 추정
> * 개인별 Insulin sensitivity와 개인의 특성 간 상관관계 파악
선형 모델을 이용해 개인의 Insulin sensitivity가 개인의 특성 (몸무게, 나이, Hb1Ac, BMI, 치료군, 성별, BMI, 인종)과 관련이 있는지 파악
> * 심박수에 따른 insulin efficiency 변화 파악
혈당의 차분에 대해 심박수와 인슐린 양을 독립 변수로 삼아 Linear Mixture Model을 구성하고, 혈당의 차분 시간을 변화시켜 가며 심박수가 어느 시점의 혈당에 가장 큰 영향을 미쳤는지 관찰
## 프로젝트 결과
>1. LMM을 이용한 Insulin 투여와 탄수화물 섭취와 혈당 변화 시간 간격 파악 
Insulin 투여가 혈당 변화에 가장 큰 영향을 미치는 시간은 투여 후 50분, 탄수화물 섭취가 가장 큰
영향을 미치는 시간은 섭취 후 35분으로 나타남
>2. Insulin sensitivity와 개인별 특성 간 상관관계 파악
Insulin Sensitivity는 LMM의 경우 몸무게, 시계열 모형의 경우 나이, BMI 지표와 약한 관련성이 있는
것으로 나타남
>3. 탄수화물 섭취와 심박수 증가, 그리고 그로 인한 인슐린의 혈당 강하 효율 파악
탄수화물 섭취는 유의미한 심박수 상승을 유발했으며, 이러한 심박수 증가는 중단기적으로 인슐린
의 신체 내 분포 및 작용 속도를 높여 혈당 강하 효과를 증대시켰음
## 향후 발전 방향
> * 계산한 개별 환자의 인슐린 민감도를 이용해 1형 당뇨병 환자에서 개인 맞춤형 인슐린 투입량 계산, 제시 가능
> * 신체 활동, 정신 상태, 시간대 등을 구체적으로 고려하면 동적 인슐린 민감도 및 상황별 개인화된 인슐린 필요량을 구체적으로 개선 가능할 것으로 보임
