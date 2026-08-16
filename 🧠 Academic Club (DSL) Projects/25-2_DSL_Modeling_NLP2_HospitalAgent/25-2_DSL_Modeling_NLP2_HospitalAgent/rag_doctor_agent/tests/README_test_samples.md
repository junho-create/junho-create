# 의료 상담 Agent 테스트 샘플 파일들

현재 의료 정보 데이터셋을 기반으로 생성된 다양한 테스트 시나리오입니다.

## 📋 테스트 케이스 목록

### 🦴 정형외과 관련
- **`sample_knee_pain.json`**: 무릎 통증, 퇴행성관절염 관련 증상
- **`sample_shoulder_pain.json`**: 어깨 통증, 오십견, 회전근개 손상 관련

### 🏥 척추/신경외과 관련  
- **`sample_back_pain.json`**: 허리디스크, 척추관협착증 관련 증상
- **`sample_neck_pain.json`**: 목디스크, 거북목 관련 증상

### 🧠 신경과/뇌신경센터 관련
- **`sample_headache.json`**: 두통, 편두통 관련 증상
- **`sample_dizziness.json`**: 어지럼증, 이석증 관련 증상

### 🏥 내과/건강검진 관련
- **`sample_endoscopy.json`**: 위내시경, 소화기 관련 증상
- **`sample_health_checkup.json`**: 종합건강검진, 암검진 관련

### 🚨 특수 상황
- **`sample_complex_symptoms.json`**: 복합적인 증상을 가진 환자
- **`sample_emergency.json`**: 응급상황, 응급의학과 관련
- **`sample_specific_doctor.json`**: 특정 의사 지정 진료 요청

## 🧪 테스트 실행 방법

### 1. 개별 테스트 실행
```bash
# 무릎 통증 케이스 테스트
python run_sample.py tests/sample_knee_pain.json

# 두통 케이스 테스트  
python run_sample.py tests/sample_headache.json

# 응급상황 케이스 테스트
python run_sample.py tests/sample_emergency.json
```

### 2. 전체 테스트 실행 (배치)
```bash
# tests 폴더의 모든 sample_*.json 파일 테스트
for file in tests/sample_*.json; do
    echo "Testing: $file"
    python run_sample.py "$file"
    echo "---"
done
```

### 3. CLI 매니저를 통한 테스트
```bash
# CLI 매니저 실행 후 테스트 파일 경로 입력
python manager_cli.py
```

## 📊 예상 테스트 결과

각 테스트 케이스는 다음을 검증합니다:

1. **증상 분석**: 환자 증상에 대한 정확한 의학적 분석
2. **진료과 추천**: 적절한 진료과 매칭
3. **의료진 추천**: 전문분야에 맞는 의료진 추천  
4. **예약 스케줄링**: 환자 선호시간과 의료진 스케줄 매칭
5. **추가 검사 제안**: 필요한 검사나 치료 방법 제안

## 💡 테스트 팁

- 각 케이스는 실제 환자가 사용할 법한 자연스러운 표현으로 작성됨
- 의료진과 증상 데이터베이스의 매칭 정확도를 확인할 수 있음
- 복합 증상이나 애매한 케이스의 처리 능력을 평가할 수 있음
- 응급상황이나 특수 요청에 대한 적절한 대응 확인 가능

## 🔧 커스텀 테스트 케이스 작성

새로운 테스트 케이스를 만들 때는 기존 파일 구조를 참고하여 작성하세요:

```json
{
  "patient_name": "환자명",
  "patient_gender": "성별",
  "phone_num": "전화번호", 
  "chat_start_date": "상담시작일시",
  "symptoms": ["증상1", "증상2", ...],
  "visit_type": "초진|재진|응급",
  "preference_datetime": ["희망일시1", "희망일시2", ...],
  "dept": "진료과|null",
  "doctor_name": "의사명|null", 
  "other_info": ["기타정보1", "기타정보2", ...]
}
```
