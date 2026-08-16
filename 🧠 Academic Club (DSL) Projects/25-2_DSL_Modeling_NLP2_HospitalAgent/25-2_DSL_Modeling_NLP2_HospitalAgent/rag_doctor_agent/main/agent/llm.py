from __future__ import annotations
from typing import List, Dict, Any, Optional
import os, json
from dotenv import load_dotenv
load_dotenv()

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

class LLMClient:
    def __init__(self, model: str = None, embed_model: str = None):
        self.chat_model = model or os.getenv("CHAT_MODEL", "gpt-4.1-mini")
        self.embed_model = embed_model or os.getenv("EMBED_MODEL", "text-embedding-3-large")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or OpenAI is None:
            raise RuntimeError("OPENAI_API_KEY missing or openai package not available.")
        self.client = OpenAI()

    def embed(self, texts: List[str]):
        # Provided for convenience; embedding is handled by embeddings_openai in retriever
        res = self.client.embeddings.create(model=self.embed_model, input=texts)
        return [d.embedding for d in res.data]

    def structured_select(self, query_payload: Dict[str, Any], retrieved: List[Dict[str, Any]], rules) -> Optional[Dict[str, Any]]:
        from ..prompts.prompt_templates import SYSTEM_PROMPT
        # TOP_K 환경변수를 고려한 동적 프롬프트 생성
        top_k = rules.get_top_k()
        dynamic_prompt = SYSTEM_PROMPT.replace("TOP K", f"TOP {top_k}")
        system = {"role": "system", "content": dynamic_prompt}
        user = {"role": "user", "content": json.dumps({"input": query_payload, "retrieved": retrieved, "rules": rules.rules, "top_k": top_k}, ensure_ascii=False)}
        tool_schema = {
            "type": "function",
            "function": {
                "name": "select_doctor",
                "description": "진료과/의료진 매핑을 구조화 JSON으로 반환",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dept": {"type": "string"},
                        "doctor_name": {"type": "string"},
                        "top_k_suggestions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "의료진명": {"type": ["string", "null"]},
                                    "진료과": {"type": ["string", "null"]},
                                    "환자의 구체적인 증상": {"type": ["array", "null"], "items": {"type": "string"}},
                                    "이유": {"type": ["string", "null"]}
                                },
                                "required": ["의료진명", "진료과", "환자의 구체적인 증상", "이유"],
                                "additionalProperties": False
                            }
                        },
                        "retrieval_evidence": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["dept", "doctor_name", "top_k_suggestions", "retrieval_evidence"],
                    "additionalProperties": False
                }
            }
        }
        try:
            res = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[system, user],
                tools=[tool_schema],
                tool_choice={"type": "function", "function": {"name": "select_doctor"}},
                temperature=0.2,
            )
            if res.choices and res.choices[0].message.tool_calls:
                tool_call = res.choices[0].message.tool_calls[0]
                if tool_call.function.name == "select_doctor":
                    try:
                        return json.loads(tool_call.function.arguments)
                    except Exception:
                        pass
        except Exception:
            return None
        return None

    def process_admin_command(self, command_text: str, current_rules: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """자연어 관리자 명령을 해석하고 실행합니다."""
        system_prompt = """당신은 병원 관리 시스템의 AI 어시스턴트입니다. 
사용자의 자연어 명령을 분석하여 다음 작업들을 수행할 수 있습니다:

1. 의료진 예약 상태 변경 (예: "김재훈 선생님 예약 중단", "이영희 교수님 예약 재개")
2. 직함 우선순위 설정 (예: "센터장 → 교수 → 전문의 순서로 우선순위 설정")
3. 진료과별 가중치 조정 (예: "내과 가중치를 1.5로 설정")
4. 증상별 매핑 규칙 추가 (예: "두통은 신경과로 연결")

명령을 분석하여 적절한 작업 타입과 매개변수를 JSON으로 반환하세요.
"""

        user_prompt = f"""
현재 규칙 상태:
{json.dumps(current_rules, ensure_ascii=False, indent=2)}

사용자 명령: "{command_text}"

이 명령을 분석하여 다음 형식으로 응답하세요:
"""

        tool_schema = {
            "type": "function",
            "function": {
                "name": "execute_admin_command",
                "description": "관리자 명령을 해석하고 실행 계획을 반환합니다",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command_type": {
                            "type": "string",
                            "enum": ["doctor_availability", "title_priority", "dept_weight", "symptom_mapping", "unknown"],
                            "description": "명령의 유형"
                        },
                        "action": {
                            "type": "string",
                            "enum": ["set", "add", "remove", "update"],
                            "description": "수행할 작업"
                        },
                        "parameters": {
                            "type": "object",
                            "description": "명령 실행에 필요한 매개변수들"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "명령 해석의 신뢰도 (0-1)"
                        },
                        "explanation": {
                            "type": "string",
                            "description": "명령 해석에 대한 설명"
                        }
                    },
                    "required": ["command_type", "action", "parameters", "confidence", "explanation"]
                }
            }
        }

        try:
            res = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=[tool_schema],
                tool_choice={"type": "function", "function": {"name": "execute_admin_command"}},
                temperature=0.1,
            )
            
            if res.choices and res.choices[0].message.tool_calls:
                tool_call = res.choices[0].message.tool_calls[0]
                if tool_call.function.name == "execute_admin_command":
                    try:
                        return json.loads(tool_call.function.arguments)
                    except Exception:
                        pass
        except Exception as e:
            # print(f"LLM 명령어 처리 오류: {e}")
            return None
        return None

    def convert_symbols_to_natural_language(self, text: str) -> Optional[str]:
        """기호가 포함된 텍스트를 자연어로 변환합니다."""
        system_prompt = """당신은 병원 관리 시스템에서 기호가 포함된 텍스트를 자연어로 변환하는 전문가입니다.
다양한 기호들을 의미에 맞는 한국어 자연어로 변환해주세요.

변환 규칙:
1. 직함 우선순위 매핑 (가장 중요!):
   - "센터장→원장→교수→전문의" 같은 패턴은 의료진 매핑 순서를 의미합니다
   - → (유니코드 화살표), - (하이픈), > 등은 모두 "다음"으로 변환
   - 반드시 끝에 "순으로 우선순위 설정"을 추가해주세요
   
2. 가중치 설정:
   - = (등호): "가중치를" 또는 "값을"
   - : (콜론): "가중치를" (숫자가 뒤에 오는 경우)
   
3. 증상-진료과 매핑:
   - -> (ASCII 화살표): "은" + "로 연결"
   - : (콜론): "은" + "로 연결" (진료과가 뒤에 오는 경우)

예시:
- "센터장→원장→교수→전문의" → "센터장 다음 원장 다음 교수 다음 전문의 순으로 우선순위 설정"
- "대표원장→교수→전문의" → "대표원장 다음 교수 다음 전문의 순으로 우선순위 설정"
- "내과 = 1.5" → "내과 가중치를 1.5로 설정"
- "두통 -> 신경과" → "두통은 신경과로 연결"
- "복통: 내과" → "복통은 내과로 연결"

특히 직함(원장, 교수, 전문의, 센터장 등)이 포함된 화살표 패턴은 반드시 매핑 순서 설정으로 해석해주세요."""

        user_prompt = f'다음 텍스트를 자연어로 변환해주세요: "{text}"'

        tool_schema = {
            "type": "function",
            "function": {
                "name": "convert_to_natural_language",
                "description": "기호가 포함된 텍스트를 자연어로 변환합니다",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "converted_text": {
                            "type": "string",
                            "description": "자연어로 변환된 텍스트"
                        },
                        "conversion_type": {
                            "type": "string",
                            "enum": ["priority_order", "weight_setting", "symptom_mapping", "general"],
                            "description": "변환 타입"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "변환의 신뢰도"
                        }
                    },
                    "required": ["converted_text", "conversion_type", "confidence"]
                }
            }
        }

        try:
            res = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=[tool_schema],
                tool_choice={"type": "function", "function": {"name": "convert_to_natural_language"}},
                temperature=0.1,
            )
            
            if res.choices and res.choices[0].message.tool_calls:
                tool_call = res.choices[0].message.tool_calls[0]
                if tool_call.function.name == "convert_to_natural_language":
                    try:
                        result = json.loads(tool_call.function.arguments)
                        if result.get("confidence", 0) > 0.7:  # 신뢰도 70% 이상만
                            return result.get("converted_text")
                    except Exception:
                        pass
        except Exception as e:
            # print(f"LLM 기호 변환 오류: {e}")
            return None
        return None
