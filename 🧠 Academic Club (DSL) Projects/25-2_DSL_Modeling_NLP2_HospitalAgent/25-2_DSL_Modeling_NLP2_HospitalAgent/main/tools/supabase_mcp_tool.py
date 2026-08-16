#!/usr/bin/env python3
"""
Supabase MCP 도구 (LangChain MCP Adapters 사용)
"""
import os
import json
from typing import Dict, Any, List, Optional
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from supabase import create_client, Client
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

class SupabaseDirectTool(BaseTool):
    """직접 Supabase 클라이언트를 사용한 도구"""
    
    name: str = "supabase_direct"
    description: str = "직접 Supabase 클라이언트를 사용한 데이터베이스 작업"
    supabase_client: Optional[Client] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_supabase_client()
    
    def _initialize_supabase_client(self):
        """Supabase 클라이언트 초기화"""
        try:
            # Supabase 설정
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_ANON_KEY')
            
            if not supabase_url or not supabase_key:
                raise ValueError("Supabase 환경 변수가 설정되지 않았습니다.")
            
            # Supabase 클라이언트 초기화
            self.supabase_client = create_client(supabase_url, supabase_key)
            
            print(f"✅ Supabase 클라이언트 초기화 완료: {supabase_url}")
            
        except Exception as e:
            print(f"❌ Supabase 클라이언트 초기화 실패: {e}")
            self.supabase_client = None
    
    def _run(
        self,
        table: str = "예약정보",
        operation: str = "select",
        filters: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """도구 실행"""
        try:
            if not self.supabase_client:
                return json.dumps({
                    "success": False,
                    "error": "Supabase 클라이언트가 초기화되지 않았습니다.",
                    "message": "Supabase 연결을 확인해주세요."
                })
            
            # Supabase 클라이언트로 직접 쿼리 실행
            if operation == "select":
                # SELECT 쿼리 실행
                query = self.supabase_client.table(table).select('*')
                if filters:
                    for key, value in filters.items():
                        query = query.eq(key, value)
                result = query.execute()
                
                return json.dumps({
                    "success": True,
                    "data": result.data,
                    "message": f"{table} 테이블에서 {len(result.data)}건의 데이터를 조회했습니다."
                })
            
            elif operation == "insert":
                # INSERT 쿼리 실행
                if not data:
                    return json.dumps({
                        "success": False,
                        "error": "삽입할 데이터가 제공되지 않았습니다.",
                        "message": "data 파라미터를 제공해주세요."
                    })
                
                result = self.supabase_client.table(table).insert(data).execute()
                
                return json.dumps({
                    "success": True,
                    "data": result.data,
                    "message": f"{table} 테이블에 데이터가 성공적으로 삽입되었습니다."
                })
            
            elif operation == "update":
                # UPDATE 쿼리 실행
                if not filters or not data:
                    return json.dumps({
                        "success": False,
                        "error": "필터 조건과 수정할 데이터가 필요합니다.",
                        "message": "filters와 data 파라미터를 제공해주세요."
                    })
                
                query = self.supabase_client.table(table).update(data)
                for key, value in filters.items():
                    query = query.eq(key, value)
                result = query.execute()
                
                return json.dumps({
                    "success": True,
                    "data": result.data,
                    "message": f"{table} 테이블의 데이터가 성공적으로 수정되었습니다."
                })
            
            elif operation == "delete":
                # DELETE 쿼리 실행
                if not filters:
                    return json.dumps({
                        "success": False,
                        "error": "삭제할 조건이 제공되지 않았습니다.",
                        "message": "filters 파라미터를 제공해주세요."
                    })
                
                query = self.supabase_client.table(table).delete()
                for key, value in filters.items():
                    query = query.eq(key, value)
                result = query.execute()
                
                return json.dumps({
                    "success": True,
                    "data": result.data,
                    "message": f"{table} 테이블에서 데이터가 성공적으로 삭제되었습니다."
                })
            
            else:
                return json.dumps({
                    "success": False,
                    "error": f"지원하지 않는 작업: {operation}",
                    "message": "select, insert, update, delete 중 하나를 선택해주세요."
                })
                
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e),
                "message": "Supabase 도구 실행 중 오류가 발생했습니다."
            })
    
    def get_available_operations(self) -> List[str]:
        """사용 가능한 작업 목록 반환"""
        if not self.supabase_client:
            return []
        
        try:
            # 사용 가능한 작업 목록 반환
            return ["select", "insert", "update", "delete"]
        except Exception as e:
            print(f"❌ 작업 목록 조회 실패: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Supabase 연결 테스트"""
        try:
            if not self.supabase_client:
                return False
            
            # 간단한 테스트 요청 (테이블 조회)
            result = self.supabase_client.table('예약정보').select('*').limit(1).execute()
            return True  # 쿼리가 성공하면 연결됨
            
        except Exception as e:
            print(f"❌ Supabase 연결 테스트 실패: {e}")
            return False

# 직접 Supabase 클라이언트를 사용한 특화 도구들
class SupabaseReadTool(SupabaseDirectTool):
    """읽기 전용 도구"""
    name: str = "supabase_read_direct"
    description: str = "직접 Supabase 클라이언트를 사용한 데이터 조회 - 예약정보, 환자정보, 의사, 가용일정, 과거상태 테이블 조회"
    
    def _run(
        self,
        table: str = "예약정보",
        filters: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """데이터 조회"""
        return super()._run(
            table=table,
            operation="select",
            filters=filters,
            run_manager=run_manager
        )

class SupabaseCreateTool(SupabaseDirectTool):
    """생성 전용 도구"""
    name: str = "supabase_create_direct"
    description: str = "직접 Supabase 클라이언트를 사용한 데이터 생성 - 예약정보, 환자정보, 가용일정 테이블 생성"
    
    def _run(
        self,
        table: str = "예약정보",
        data: Dict[str, Any] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """데이터 생성"""
        if data is None:
            data = {}
        
        return super()._run(
            table=table,
            operation="insert",
            data=data,
            run_manager=run_manager
        )

class SupabaseUpdateTool(SupabaseDirectTool):
    """수정 전용 도구"""
    name: str = "supabase_update_direct"
    description: str = "직접 Supabase 클라이언트를 사용한 데이터 수정 - 예약정보, 환자정보, 가용일정 테이블 수정"
    
    def _run(
        self,
        table: str = "예약정보",
        filters: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """데이터 수정"""
        return super()._run(
            table=table,
            operation="update",
            filters=filters,
            data=data,
            run_manager=run_manager
        )

class SupabaseDeleteTool(SupabaseDirectTool):
    """삭제 전용 도구"""
    name: str = "supabase_delete_direct"
    description: str = "직접 Supabase 클라이언트를 사용한 데이터 삭제 - 예약정보, 환자정보, 가용일정 테이블 삭제"
    
    def _run(
        self,
        table: str = "예약정보",
        filters: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """데이터 삭제"""
        return super()._run(
            table=table,
            operation="delete",
            filters=filters,
            run_manager=run_manager
        )

class SupabasePatientLookupTool(SupabaseDirectTool):
    """환자 조회 전용 도구"""
    name: str = "supabase_patient_lookup"
    description: str = "전화번호로 환자정보에서 환자ID 조회 - 예약 전 환자 확인용"
    
    def _run(
        self,
        phone_number: str = "",
        patient_name: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """전화번호로 환자 조회"""
        try:
            if not self.supabase_client:
                return json.dumps({
                    "success": False,
                    "error": "Supabase 클라이언트가 초기화되지 않았습니다.",
                    "message": "Supabase 연결을 확인해주세요."
                })
            
            # 전화번호로 환자 조회
            if phone_number:
                result = self.supabase_client.table("환자정보").select("*").eq("전화번호", phone_number).execute()
            elif patient_name:
                result = self.supabase_client.table("환자정보").select("*").eq("이름", patient_name).execute()
            else:
                return json.dumps({
                    "success": False,
                    "error": "전화번호 또는 환자명이 필요합니다.",
                    "message": "phone_number 또는 patient_name을 제공해주세요."
                })
            
            return json.dumps({
                "success": True,
                "data": result.data,
                "message": f"환자 조회 완료: {len(result.data)}건"
            })
            
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e),
                "message": "환자 조회 중 오류가 발생했습니다."
            })

class SupabaseDoctorLookupTool(SupabaseDirectTool):
    """의사 조회 전용 도구"""
    name: str = "supabase_doctor_lookup"
    description: str = "의료진명으로 의사 테이블에서 DocID 조회 - 예약 일정 조회용"
    
    def _run(
        self,
        doctor_name: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """의료진명으로 의사 조회"""
        try:
            if not self.supabase_client:
                return json.dumps({
                    "success": False,
                    "error": "Supabase 클라이언트가 초기화되지 않았습니다.",
                    "message": "Supabase 연결을 확인해주세요."
                })
            
            if not doctor_name:
                return json.dumps({
                    "success": False,
                    "error": "의료진명이 필요합니다.",
                    "message": "doctor_name을 제공해주세요."
                })
            
            # 의료진명으로 의사 조회
            result = self.supabase_client.table("의사").select("*").eq("의료진명", doctor_name).execute()
            
            return json.dumps({
                "success": True,
                "data": result.data,
                "message": f"의사 조회 완료: {len(result.data)}건"
            })
            
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e),
                "message": "의사 조회 중 오류가 발생했습니다."
            })

class SupabaseScheduleLookupTool(SupabaseDirectTool):
    """가용일정 조회 전용 도구"""
    name: str = "supabase_schedule_lookup"
    description: str = "DocID로 가용일정에서 예약 가능한 일정 조회"
    
    def _run(
        self,
        doc_id: int = 0,
        limit: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """DocID로 가용일정 조회"""
        try:
            if not self.supabase_client:
                return json.dumps({
                    "success": False,
                    "error": "Supabase 클라이언트가 초기화되지 않았습니다.",
                    "message": "Supabase 연결을 확인해주세요."
                })
            
            if not doc_id:
                return json.dumps({
                    "success": False,
                    "error": "DocID가 필요합니다.",
                    "message": "doc_id를 제공해주세요."
                })
            
            # DocID로 가용일정 조회 (예약 가능한 것만)
            result = self.supabase_client.table("가용일정").select("*").eq("DocID_응급실포함", doc_id).eq("예약가능여부", "Y").limit(limit).execute()
            
            return json.dumps({
                "success": True,
                "data": result.data,
                "message": f"가용일정 조회 완료: {len(result.data)}건"
            })
            
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e),
                "message": "가용일정 조회 중 오류가 발생했습니다."
            })

# 도구 팩토리 함수
def create_supabase_direct_tools() -> List[BaseTool]:
    """직접 Supabase 클라이언트를 사용한 도구들 생성"""
    tools = []
    
    try:
        # 기본 도구
        tools.append(SupabaseDirectTool())
        
        # 특화 도구들
        tools.append(SupabaseReadTool())
        tools.append(SupabaseCreateTool())
        tools.append(SupabaseUpdateTool())
        tools.append(SupabaseDeleteTool())
        tools.append(SupabasePatientLookupTool())
        tools.append(SupabaseDoctorLookupTool())
        tools.append(SupabaseScheduleLookupTool())
        
        print(f"✅ 직접 Supabase 도구 {len(tools)}개 생성 완료")
        
    except Exception as e:
        print(f"❌ 직접 Supabase 도구 생성 실패: {e}")
    
    return tools

def get_supabase_tools_for_binding() -> List[BaseTool]:
    """bind_tools용 Supabase 도구들 반환 (중복 제거)"""
    tools = []
    
    try:
        # 특화 도구들만 반환 (기본 도구 제외)
        tools.extend([
            SupabaseReadTool(),
            SupabaseCreateTool(),
            SupabaseUpdateTool(),
            SupabaseDeleteTool(),
            SupabasePatientLookupTool(),
            SupabaseDoctorLookupTool(),
            SupabaseScheduleLookupTool()
        ])
        
        print(f"✅ bind_tools용 Supabase 도구 {len(tools)}개 준비 완료")
        
    except Exception as e:
        print(f"❌ bind_tools용 도구 생성 실패: {e}")
    
    return tools

# 테스트 함수
def test_langchain_mcp_tools():
    """LangChain MCP Adapters 도구 테스트"""
    print("🧪 LangChain MCP Adapters 도구 테스트")
    print("=" * 50)
    
    # 도구 생성
    tools = create_supabase_langchain_mcp_tools()
    
    if not tools:
        print("❌ 도구 생성 실패")
        return
    
    # 기본 도구 테스트
    basic_tool = tools[0]
    print(f"✅ 기본 도구: {basic_tool.name}")
    
    # 연결 테스트
    if basic_tool.test_connection():
        print("✅ MCP 서버 연결 성공")
    else:
        print("❌ MCP 서버 연결 실패")
        return
    
    # 사용 가능한 도구 목록
    available_tools = basic_tool.get_available_tools()
    print(f"📋 사용 가능한 도구: {available_tools}")
    
    # 읽기 도구 테스트
    read_tool = tools[1]
    print(f"✅ 읽기 도구: {read_tool.name}")
    
    # 스키마 조회 테스트
    try:
        result = basic_tool._run(
            query="스키마 조회",
            tool="supabase_get_schema",
            parameters={}
        )
        result_data = json.loads(result)
        if result_data.get('success'):
            print("✅ 스키마 조회 성공")
        else:
            print(f"❌ 스키마 조회 실패: {result_data.get('error')}")
    except Exception as e:
        print(f"❌ 스키마 조회 테스트 오류: {e}")
    
    print("\n🎉 LangChain MCP Adapters 도구 테스트 완료!")

if __name__ == "__main__":
    test_langchain_mcp_tools()
