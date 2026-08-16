"""
Tavily 웹검색 툴
바른마디병원 웹사이트(https://barunjoint.kr/)에서만 검색하도록 제한
API 사용량 한도 초과 시 정적 데이터 기반 검색으로 대체
"""
import os
import requests
from typing import Dict, List, Any
import json
from datetime import datetime

class TavilySearchTool:
    """바른마디병원 웹사이트 전용 검색 툴"""
    
    def __init__(self):
        self.base_url = "https://barunjoint.kr"
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.static_data = self._load_static_data()
        
    def _load_static_data(self) -> Dict[str, Any]:
        """정적 병원 정보 데이터 로드"""
        try:
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, "..", "..", "data", "hospital_info.json")
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"정적 데이터 로드 실패: {e}")
            return {}
    
    def search_hospital_info(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """
        바른마디병원 웹사이트에서 정보 검색
        API 사용량 한도 초과 시 정적 데이터 기반 검색으로 대체
        
        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            
        Returns:
            검색 결과 딕셔너리
        """
        # 먼저 Tavily API 시도
        if self.api_key:
            try:
                # Tavily API 호출 (barunjoint.kr 도메인으로 제한)
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                payload = {
                    "query": query,
                    "search_depth": "basic",
                    "include_answer": True,
                    "include_images": False,
                    "include_raw_content": False,
                    "max_results": max_results,
                    "include_domains": ["barunjoint.kr"],
                    "exclude_domains": []
                }
                
                response = requests.post(
                    "https://api.tavily.com/search",
                    headers=headers,
                    json=payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return self._format_search_results(result, query)
                elif response.status_code == 432:  # 사용량 한도 초과
                    print("⚠️ Tavily API 사용량 한도 초과 - 정적 데이터로 대체")
                    return self._search_static_data(query)
                else:
                    print(f"⚠️ Tavily API 오류 ({response.status_code}) - 정적 데이터로 대체")
                    return self._search_static_data(query)
                    
            except Exception as e:
                print(f"⚠️ Tavily API 호출 실패 - 정적 데이터로 대체: {e}")
                return self._search_static_data(query)
        else:
            print("⚠️ Tavily API 키 없음 - 정적 데이터로 대체")
            return self._search_static_data(query)
    
    def _format_search_results(self, raw_results: Dict, query: str) -> Dict[str, Any]:
        """검색 결과를 사용자 친화적 형식으로 포맷팅"""
        try:
            results = raw_results.get("results", [])
            answer = raw_results.get("answer", "")
            
            if not results and not answer:
                return {
                    "success": False,
                    "message": f"'{query}'에 대한 정보를 바른마디병원 웹사이트에서 찾을 수 없습니다.",
                    "suggestion": "전화 상담(1599.0015)을 이용해주세요."
                }
            
            # 검색 결과 포맷팅
            formatted_results = []
            for result in results[:3]:  # 상위 3개만
                formatted_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", ""),
                    "relevance_score": result.get("score", 0)
                })
            
            return {
                "success": True,
                "query": query,
                "answer": answer,
                "results": formatted_results,
                "source": "바른마디병원 공식 웹사이트 (barunjoint.kr)",
                "total_results": len(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "검색 결과 처리 중 오류가 발생했습니다."
            }
    
    def _search_static_data(self, query: str) -> Dict[str, Any]:
        """정적 데이터에서 병원 정보 검색"""
        try:
            if not self.static_data:
                return {
                    "success": False,
                    "message": "정적 데이터를 로드할 수 없습니다.",
                    "suggestion": "전화 상담(1599.0015)을 이용해주세요."
                }
            
            query_lower = query.lower()
            results = []
            answer_parts = []
            
            # 휴무일 검색
            if any(keyword in query_lower for keyword in ['휴무', '휴진', '휴일', '닫', '안열']):
                holidays = self.static_data.get('holidays', {})
                regular_holidays = holidays.get('regular_holidays', [])
                national_holidays = holidays.get('national_holidays', [])
                
                if regular_holidays:
                    answer_parts.append(f"정기 휴무일: {', '.join(regular_holidays)}")
                if national_holidays:
                    answer_parts.append(f"공휴일: {', '.join(national_holidays[:3])}...")
                
                results.append({
                    "title": "바른마디병원 휴무일 안내",
                    "content": f"정기 휴무일: {', '.join(regular_holidays)}\n공휴일: {', '.join(national_holidays[:5])}",
                    "url": "https://barunjoint.kr",
                    "relevance_score": 0.9
                })
            
            # 운영시간 검색
            elif any(keyword in query_lower for keyword in ['시간', '운영', '진료', '열', '오픈']):
                hours = self.static_data.get('operating_hours', {})
                weekdays = hours.get('weekdays', {})
                lunch_break = hours.get('lunch_break', '')
                
                weekday_hours = list(weekdays.values())[0] if weekdays else "09:00-18:00"
                answer_parts.append(f"평일 진료시간: {weekday_hours}")
                if lunch_break:
                    answer_parts.append(f"점심시간: {lunch_break}")
                
                results.append({
                    "title": "바른마디병원 진료시간 안내",
                    "content": f"평일: {weekday_hours}\n점심시간: {lunch_break}\n토요일: 09:00-13:00\n일요일: 휴무",
                    "url": "https://barunjoint.kr",
                    "relevance_score": 0.9
                })
            
            # 연락처 검색
            elif any(keyword in query_lower for keyword in ['전화', '연락', '번호', '문의']):
                contact = self.static_data.get('contact', {})
                phone = contact.get('phone', '1599.0015')
                address = contact.get('address', '')
                
                answer_parts.append(f"전화번호: {phone}")
                if address:
                    answer_parts.append(f"주소: {address}")
                
                results.append({
                    "title": "바른마디병원 연락처 정보",
                    "content": f"전화번호: {phone}\n주소: {address}",
                    "url": "https://barunjoint.kr",
                    "relevance_score": 0.9
                })
            
            # 진료과 검색
            elif any(keyword in query_lower for keyword in ['과', '진료과', '부서', '센터']):
                departments = self.static_data.get('departments', {})
                dept_list = []
                for dept_key, dept_info in departments.items():
                    dept_name = dept_info.get('name', dept_key)
                    specialties = dept_info.get('specialties', [])
                    dept_list.append(f"{dept_name}: {', '.join(specialties)}")
                
                answer_parts.append(f"진료과: {', '.join([dept.split(':')[0] for dept in dept_list])}")
                
                results.append({
                    "title": "바른마디병원 진료과 안내",
                    "content": f"진료과: {', '.join(dept_list)}",
                    "url": "https://barunjoint.kr",
                    "relevance_score": 0.8
                })
            
            # 일반 정보 검색
            else:
                hospital_name = self.static_data.get('hospital_name', '바른마디병원')
                contact = self.static_data.get('contact', {})
                phone = contact.get('phone', '1599.0015')
                
                answer_parts.append(f"{hospital_name}에 대한 정보입니다.")
                answer_parts.append(f"전화번호: {phone}")
                
                results.append({
                    "title": f"{hospital_name} 기본 정보",
                    "content": f"전화번호: {phone}\n웹사이트: https://barunjoint.kr",
                    "url": "https://barunjoint.kr",
                    "relevance_score": 0.7
                })
            
            return {
                "success": True,
                "query": query,
                "answer": " | ".join(answer_parts),
                "results": results,
                "source": "바른마디병원 정적 데이터",
                "total_results": len(results),
                "fallback": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "정적 데이터 검색 중 오류가 발생했습니다.",
                "suggestion": "전화 상담(1599.0015)을 이용해주세요."
            }

# LangGraph 툴로 사용할 수 있는 함수들
def search_hospital_hours(query: str = "") -> str:
    """병원 운영시간 관련 정보 검색"""
    tool = TavilySearchTool()
    search_query = f"진료시간 운영시간 {query}" if query else "진료시간 운영시간"
    result = tool.search_hospital_info(search_query)
    return _format_tool_response(result, "진료시간 정보")

def search_hospital_holidays(query: str = "") -> str:
    """병원 휴무일 관련 정보 검색"""
    tool = TavilySearchTool()
    search_query = f"휴무일 휴진일 {query}" if query else "휴무일 휴진일"
    result = tool.search_hospital_info(search_query)
    return _format_tool_response(result, "휴무일 정보")

def search_hospital_departments(query: str = "") -> str:
    """병원 진료과 관련 정보 검색"""
    tool = TavilySearchTool()
    search_query = f"진료과 센터 {query}" if query else "진료과 센터"
    result = tool.search_hospital_info(search_query)
    return _format_tool_response(result, "진료과 정보")

def search_hospital_contact(query: str = "") -> str:
    """병원 연락처 관련 정보 검색"""
    tool = TavilySearchTool()
    search_query = f"연락처 전화번호 {query}" if query else "연락처 전화번호"
    result = tool.search_hospital_info(search_query)
    return _format_tool_response(result, "연락처 정보")

def search_hospital_general(query: str) -> str:
    """일반적인 병원 정보 검색"""
    tool = TavilySearchTool()
    result = tool.search_hospital_info(query)
    return _format_tool_response(result, "병원 정보")

def _format_tool_response(result: Dict[str, Any], info_type: str) -> str:
    """툴 응답을 문자열로 포맷팅"""
    if not result.get("success"):
        return f"❌ {info_type} 검색 실패\n{result.get('message', '알 수 없는 오류가 발생했습니다.')}"
    
    response = f"📋 **{info_type} 검색 결과**\n\n"
    
    if result.get("answer"):
        response += f"**요약:**\n{result['answer']}\n\n"
    
    if result.get("results"):
        response += "**상세 정보:**\n"
        for i, res in enumerate(result["results"], 1):
            response += f"{i}. **{res['title']}**\n"
            response += f"   {res['content']}\n"
            response += f"   🔗 [자세히 보기]({res['url']})\n\n"
    
    response += f"📌 **출처:** {result.get('source', '바른마디병원 공식 웹사이트')}\n"
    response += f"📞 **추가 문의:** 1599.0015"
    
    return response
