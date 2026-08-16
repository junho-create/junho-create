"""
LangGraph Studio UI에서 사용할 메인 그래프
Chat 기능을 위한 messages 키가 포함된 그래프
"""
from main.langgraph_workflow import create_hospital_reservation_workflow
from main.langgraph_state import HospitalReservationState
from langchain_core.messages import HumanMessage, AIMessage

# LangGraph Studio UI에서 사용할 그래프
graph = create_hospital_reservation_workflow()

# LangGraph Studio UI에서 인식할 수 있도록 graph 변수로 export
__all__ = ["graph"]
