"""
에이전트3: RAG 기반 증상-의료진 매핑
기존 rag_doctor_agent를 활용하여 증상을 받아 적절한 의료진을 추천
"""
import sys
import os
from typing import Dict, List, Any, Optional

# 기존 rag_doctor_agent 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_agent_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "rag_doctor_agent")
sys.path.append(rag_agent_path)

class Agent3RAG:
    """RAG 기반 증상-의료진 매핑 에이전트"""
    
    def __init__(self):
        self.rag_pipeline = None
        self._initialize_rag_pipeline()
    
    def _initialize_rag_pipeline(self):
        """기존 RAG 파이프라인 초기화"""
        try:
            # rag_doctor_agent 폴더에서 import
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '../../../rag_doctor_agent'))
            
            from a2a_wrapper import rag_wrapper
            self.rag_pipeline = rag_wrapper
            print("✅ RAG 파이프라인 초기화 완료")
        except ImportError as e:
            print(f"⚠️ RAG 파이프라인 초기화 실패: {e}")
            print("기본 매핑 로직을 사용합니다.")
            self.rag_pipeline = None
        except Exception as e:
            print(f"❌ RAG 파이프라인 초기화 오류: {e}")
            self.rag_pipeline = None
    
    def recommend_doctors(self, symptoms: List[str], additional_info: str = "") -> Dict[str, Any]:
        """
        증상을 기반으로 적절한 의료진 추천
        
        Args:
            symptoms: 증상 리스트
            additional_info: 추가 정보 (선호 시간, 진료과 등)
            
        Returns:
            추천 의료진 정보
        """
        try:
            if not symptoms:
                return {
                    "success": False,
                    "error": "증상 정보가 제공되지 않았습니다.",
                    "message": "어떤 증상이 있으신지 알려주세요."
                }
            
            # RAG 파이프라인이 있으면 사용, 없으면 기본 로직 사용
            if self.rag_pipeline:
                return self._rag_based_recommendation(symptoms, additional_info)
            else:
                return self._rule_based_recommendation(symptoms, additional_info)
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "의료진 추천 중 오류가 발생했습니다."
            }
    
    def _rag_based_recommendation(self, symptoms: List[str], additional_info: str) -> Dict[str, Any]:
        """RAG 파이프라인을 사용한 추천"""
        try:
            # RAG 에이전트 직접 호출
            from rag_doctor_agent.main.agent.graph import build_and_run_agent
            
            input_data = {
                "symptoms": symptoms,
                "additional_info": additional_info,
                "query": f"증상: {', '.join(symptoms)}. {additional_info}"
            }
            
            print(f"🔍 RAG 에이전트 호출: {input_data}")
            
            # RAG 에이전트 실행
            result = build_and_run_agent(input_data)
            
            print(f"📝 RAG 에이전트 결과: {result}")
            
            # 결과 파싱
            if result:
                # result가 Pydantic 모델인 경우
                if hasattr(result, 'model_dump'):
                    result_dict = result.model_dump(by_alias=True)
                else:
                    result_dict = result
                
                # RAG 결과에서 추천 의료진 추출
                recommended_doctors = []
                if "top_k_suggestions" in result_dict:
                    for suggestion in result_dict["top_k_suggestions"]:
                        if suggestion and suggestion.get("의료진명") and suggestion.get("의료진명") != "None":
                            recommended_doctors.append({
                                "name": suggestion.get("의료진명"),
                                "department": suggestion.get("진료과"),
                                "reasoning": suggestion.get("이유", ""),
                                "symptoms": suggestion.get("환자의 구체적인 증상", [])
                            })
                
                # 진료과 추출
                department = result_dict.get("dept", "")
                
                # 신뢰도 계산 (추천 의료진 수 기반)
                confidence = min(0.9, 0.5 + (len(recommended_doctors) * 0.1))
                
                return {
                    "success": True,
                    "recommended_doctors": recommended_doctors,
                    "department": department,
                    "confidence": confidence,
                    "reasoning": f"RAG 파이프라인을 통한 {len(recommended_doctors)}명의 의료진 추천",
                    "source": "RAG 파이프라인"
                }
            else:
                # RAG 실패 시 기본 로직으로 폴백
                return self._rule_based_recommendation(symptoms, additional_info)
                
        except Exception as e:
            print(f"RAG 파이프라인 오류: {e}")
            import traceback
            traceback.print_exc()
            return self._rule_based_recommendation(symptoms, additional_info)
    
    def _rule_based_recommendation(self, symptoms: List[str], additional_info: str) -> Dict[str, Any]:
        """규칙 기반 추천 (RAG 실패 시 폴백)"""
        
        # 증상별 진료과 매핑 규칙
        symptom_to_department = {
            # 관절 관련
            "무릎": "정형외과", "어깨": "정형외과", "팔꿈치": "정형외과", 
            "발목": "정형외과", "손목": "정형외과", "허리": "정형외과",
            "관절": "정형외과", "통증": "정형외과", "부상": "정형외과",
            
            # 척추 관련
            "목": "척추센터", "등": "척추센터", "허리": "척추센터",
            "디스크": "척추센터", "척추": "척추센터", "요통": "척추센터",
            
            # 내과 관련
            "복통": "내과", "소화": "내과", "위": "내과", "장": "내과",
            "내시경": "내과", "검진": "내과", "비만": "내과",
            "당뇨": "내과", "고혈압": "내과", "고지혈": "내과",
            
            # 뇌신경 관련
            "두통": "뇌신경센터", "어지럼": "뇌신경센터", "신경": "뇌신경센터",
            "치매": "뇌신경센터", "뇌졸중": "뇌신경센터", "뇌": "뇌신경센터",
            
            # 응급
            "응급": "응급의학센터", "긴급": "응급의학센터", "사고": "응급의학센터"
        }
        
        # 증상 분석
        symptoms_text = " ".join(symptoms).lower()
        matched_departments = []
        
        for symptom, department in symptom_to_department.items():
            if symptom in symptoms_text:
                if department not in matched_departments:
                    matched_departments.append(department)
        
        # 기본 진료과 설정
        if not matched_departments:
            primary_department = "내과"  # 기본값
            confidence = 0.5
        else:
            primary_department = matched_departments[0]
            confidence = min(0.9, 0.6 + len(matched_departments) * 0.1)
        
        # 가상의 의료진 추천 (실제로는 데이터베이스에서 조회)
        recommended_doctors = self._get_doctors_by_department(primary_department)
        
        return {
            "success": True,
            "recommended_doctors": recommended_doctors,
            "department": primary_department,
            "confidence": confidence,
            "reasoning": f"증상 '{', '.join(symptoms)}'에 따라 {primary_department}을 추천합니다.",
            "source": "규칙 기반 매핑",
            "alternative_departments": matched_departments[1:] if len(matched_departments) > 1 else []
        }
    
    def _get_doctors_by_department(self, department: str) -> List[Dict[str, Any]]:
        """진료과별 의료진 목록 (가상 데이터)"""
        
        doctors_database = {
            "정형외과": [
                {"name": "김정형", "specialty": "무릎관절", "experience": "15년", "rating": 4.8},
                {"name": "이어깨", "specialty": "어깨관절", "experience": "12년", "rating": 4.7},
                {"name": "박척추", "specialty": "척추정형", "experience": "18년", "rating": 4.9}
            ],
            "척추센터": [
                {"name": "최척추", "specialty": "척추수술", "experience": "20년", "rating": 4.9},
                {"name": "정디스크", "specialty": "디스크치료", "experience": "14년", "rating": 4.6},
                {"name": "한척추", "specialty": "비수술치료", "experience": "16년", "rating": 4.7}
            ],
            "내과": [
                {"name": "김내과", "specialty": "소화기내과", "experience": "22년", "rating": 4.8},
                {"name": "이검진", "specialty": "건강검진", "experience": "10년", "rating": 4.5},
                {"name": "박내시경", "specialty": "내시경검사", "experience": "15년", "rating": 4.7}
            ],
            "뇌신경센터": [
                {"name": "김뇌신경", "specialty": "두통클리닉", "experience": "18년", "rating": 4.8},
                {"name": "이어지럼", "specialty": "어지럼증", "experience": "13년", "rating": 4.6},
                {"name": "박신경", "specialty": "신경통증", "experience": "16년", "rating": 4.7}
            ],
            "응급의학센터": [
                {"name": "김응급", "specialty": "응급의학", "experience": "12년", "rating": 4.8},
                {"name": "이외상", "specialty": "외상치료", "experience": "14년", "rating": 4.7}
            ]
        }
        
        return doctors_database.get(department, [
            {"name": "김의사", "specialty": "일반진료", "experience": "10년", "rating": 4.5}
        ])

# LangGraph 노드에서 사용할 수 있는 함수
def recommend_doctors_for_symptoms(symptoms: List[str], additional_info: str = "") -> Dict[str, Any]:
    """증상에 따른 의료진 추천 (LangGraph 노드용)"""
    agent = Agent3RAG()
    return agent.recommend_doctors(symptoms, additional_info)

def format_doctor_recommendation(result: Dict[str, Any]) -> str:
    """의료진 추천 결과를 사용자 친화적 형식으로 포맷팅"""
    if not result.get("success"):
        return f"❌ 의료진 추천 실패\n{result.get('message', '알 수 없는 오류가 발생했습니다.')}"
    
    recommended_doctors = result.get("recommended_doctors", [])
    department = result.get("department", "")
    confidence = result.get("confidence", 0)
    reasoning = result.get("reasoning", "")
    
    if not recommended_doctors:
        return f"📋 **추천 진료과**: {department}\n\n{reasoning}\n\n해당 진료과의 의료진 정보를 확인 중입니다.\n📞 추가 문의: 1599.0015"
    
    response = f"👨‍⚕️ **의료진 추천 결과**\n\n"
    response += f"📋 **추천 진료과**: {department}\n"
    response += f"🎯 **추천 근거**: {reasoning}\n"
    response += f"📊 **신뢰도**: {confidence:.1%}\n\n"
    
    response += "**추천 의료진:**\n"
    for i, doctor in enumerate(recommended_doctors[:3], 1):  # 상위 3명만
        response += f"{i}. **{doctor['name']} 의사**\n"
        response += f"   • 전문분야: {doctor['specialty']}\n"
        response += f"   • 경력: {doctor['experience']}\n"
        response += f"   • 평점: ⭐ {doctor['rating']}\n\n"
    
    if len(recommended_doctors) > 3:
        response += f"... 외 {len(recommended_doctors) - 3}명의 의료진이 더 있습니다.\n\n"
    
    response += "📅 **다음 단계**: 예약을 원하시면 의료진을 선택하고 예약 정보를 알려주세요!\n"
    response += "📞 **추가 문의**: 1599.0015"
    
    return response
