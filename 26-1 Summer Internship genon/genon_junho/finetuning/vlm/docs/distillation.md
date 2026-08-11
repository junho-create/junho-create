# VLM 모델 증류 가이드

1) Distillation이란 무엇인가

**Teacher(큰/좋은 모델)**의 동작을 **Student(작은/싼 모델)**이 최대한 비슷하게 하도록 학습시키는 방법입니다.

핵심은 이거예요:
	•	사람 GT(label) 대신(또는 함께)
	•	Teacher가 만들어낸 출력/확률/중간표현을 “정답”처럼 삼아
	•	Student가 Teacher의 “함수”를 근사하게 함

⸻

2) Distillation에 필요한 데이터는?
	•	입력 데이터 x는 반드시 필요합니다. (Teacher도 입력이 있어야 출력을 만듦)
	•	사람이 만든 레이블(GT)은 없어도 가능합니다.
(Teacher 출력이 pseudo-label 역할을 함)
	•	가장 강력한 구성은 보통:
	•	소량 GT + 대량 Teacher label(증류)

⸻

3) Distillation 학습 방식의 큰 분류 (지도 신호가 뭐냐)

아래 6가지가 “증류 방식”의 대표 카테고리입니다.

A. Response(출력) Distillation — 가장 흔함

Teacher가 생성한 “최종 텍스트”를 정답처럼 두고 Student를 SFT로 학습합니다.
	•	손실: 일반적으로 Cross Entropy(CE) (정답 토큰 맞추기)
	•	장점: 구현 쉬움, 대량 생성 쉬움, 실무 최다 사용
	•	단점: Teacher의 **확률분포(soft 정보)**는 사라짐 → “진짜 KD”보단 약함

✅ 테이블 구조(HTML/JSON) 같은 구조 출력에 특히 많이 씁니다.

⸻

B. Logit / Soft-label Distillation — “진짜 KD”에 가까움

Teacher의 토큰 확률분포(로그릿)를 Student가 따라가게 합니다.
	•	손실: KL divergence
	•	KL(P_teacher || P_student)
	•	보통 temperature(τ)로 분포를 부드럽게 함
	•	장점: 작은 모델이 Teacher의 “감각(2순위 후보 등)”을 배움 → 성능/수렴 좋아지는 경우 많음
	•	단점: Teacher logits 저장/전달 비용이 큼(토큰마다 vocab 크기)

실무에서 타협:
	•	전체 vocab logits 대신 top-k logprobs만 저장
	•	또는 온라인 호출로 logits 받아오되(느림/불안정)

⸻

C. Hybrid: GT + KD 혼합 — 성능 상한 올리는 정석

사람 GT가 있을 때 가장 추천되는 형태.
	•	손실:
	•	L = α * L_GT(CE) + (1-α) * L_KD(KL)
	•	장점:
	•	GT로 정답 기준 고정
	•	KD로 학습 부드럽게/일반화
	•	단점: 데이터 파이프라인이 2종 신호를 다뤄야 함

✅ “Teacher 오류 복제” 리스크를 줄이는 가장 안전한 조합.

⸻

D. Feature / Representation Distillation — 중간표현을 맞춤

Teacher의 중간 레이어 hidden state/attention map 등을 Student가 맞추도록 합니다.
	•	예: 특정 레이어의 hidden을 MSE로 맞춤
	•	장점: 출력만 맞추는 것보다 “내부 계산”까지 유도 → 데이터 효율 좋아질 때가 있음
	•	단점:
	•	Teacher/Student 구조가 달라서 레이어 매칭이 까다로움
	•	LLM/VLM에서는 운영 난이도↑

✅ 비전 쪽(ViT, CNN)에서는 꽤 흔하고, LLM에서도 연구/특정 케이스에 쓰입니다.

⸻

E. Reasoning / CoT Distillation — “생각 과정”까지 배우게

Teacher(Thinking 모델)가 **추론 과정(Chain-of-Thought)**을 출력하면, Student가 그것까지 학습합니다.
	•	형태:
	1.	reasoning + final answer를 그대로 학습
	2.	reasoning은 요약/구조화해서 학습(더 안전/짧음)
	•	장점:
	•	Student가 “정답만”이 아니라 판단 논리/절차를 배움
	•	특히 구조 판단(표 span 등)에 도움될 수 있음
	•	단점:
	•	reasoning이 길면 학습/토큰 비용 폭증
	•	teacher의 나쁜 습관(헛추론)도 복제 가능
	•	공개 서비스에서는 reasoning 노출 정책 이슈도 생길 수 있음

✅ 테이블 구조에선 “셀 병합 판단 근거” 같은 짧은 구조 reasoning만 넣는 방식이 실용적입니다.

⸻

F. Preference Distillation (DPO/RLHF 스타일) — “좋은 답을 고르게”

Teacher가 여러 후보 중 선호를 제공하거나, GT/규칙으로 선호를 만들고 Student가 선호를 학습합니다.
	•	예: DPO/ORPO 형태
	•	장점: “미세한 품질 차이” (스타일, 안정성, 일관성) 개선
	•	단점: 구조 정확도(특히 rowspan/colspan 같은) 자체를 0→1로 만들기엔 한계가 큼

✅ 보통 SFT(또는 KD)로 기본기 만든 뒤, 마지막 다듬기로 씁니다.

⸻

4) 운영 방식 기준 분류 (프로세스가 뭐냐)

1) Offline Distillation (추천)
	1.	Teacher 서버/배치로 라벨 생성
	2.	결과를 저장(필터링/검증)
	3.	Student는 그 데이터로 학습

	•	장점: 학습이 안정적, 재현 가능, 비용 예측 쉬움
	•	단점: 라벨 생성 파이프라인 별도 필요

✅ 대부분의 기업 환경에서 이 방식이 정답입니다.

⸻

2) Online Distillation (학습 중 Teacher 실시간 호출)

학습 step마다 Teacher API를 호출해 라벨/로짓을 받아옴
	•	장점: logits/soft KD를 즉시 활용하기 쉬움
	•	단점: 느림, 장애/지연이 학습을 망침, 비용 폭발

✅ 연구/소규모 실험엔 가능하지만, 대규모 학습엔 보통 비추.

⸻

5) 구조적으로 더 세분화하면 (실무에 자주 나오는 패턴)
	•	Self-distillation: 같은 모델(또는 과거 체크포인트)을 Teacher로
	•	Multi-teacher: 여러 Teacher(예: OCR 특화 + 추론 특화)를 혼합
	•	Progressive distillation: 235B → 32B → 9B처럼 단계적으로 줄임
	•	Curriculum distillation: 쉬운 샘플→어려운 샘플 순으로
	•	Hard-case replay: 실패 케이스만 모아 재학습 (가성비 최고)

⸻

6) “어떤 증류 방식을 선택해야 하냐” 빠른 가이드

테이블 구조(HTML/JSON) 목표라면 추천 순서
	1.	**Response distillation(SFT)**로 포맷 안정성/기본 구조부터 잡기
	2.	하드케이스 리플레이로 span 오류 줄이기
	3.	필요하면 (가능 범위에서) logit KD 또는 짧은 reasoning distill 추가
	4.	마지막 다듬기로 선호학습(DPO류) 고려

⸻

7) 한 장 요약
	•	증류는 “Teacher 행동을 Student가 따라하게 만드는 학습”
	•	레이블이 없어도 가능하지만, GT가 있으면 섞는 게 가장 강력
	•	학습 방식 종류(핵심 6개):
	•	Response / Logit(KD) / Hybrid(GT+KD) / Feature / Reasoning / Preference
	•	운영 방식:
	•	Offline(추천) vs Online(비추천이 많음)
