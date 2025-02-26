# recommendation_model.py
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langgraph.checkpoint.memory import MemorySaver  
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

class RecommendationModel:
    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        language: str
        user_info: dict

    def __init__(self, model_name="gpt-4o-mini", model_provider="openai"):
        # 모델 초기화
        self.model = init_chat_model(model_name, model_provider=model_provider)
        self.llm = ChatOpenAI(model_name=model_name)

        # 사용자 정보와 대화 히스토리를 반영한 추천 프롬프트 템플릿
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "당신은 패션 및 라이프스타일 추천 어시스턴트입니다. 사용자의 생년월일, 성별, MBTI, \
                그리고 운세(추후 API 연동 예정) 등의 정보를 바탕으로 개인 맞춤형 추천을 제공하세요. 답변은 {language}로 해주세요."
            ),
            ("system", "사용자 정보: {user_info}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        # 대화 히스토리 및 메모리 설정 (SQLite 저장)
        self.memory = MemorySaver()
        self.chat_message_history = SQLChatMessageHistory(
            session_id="recommend_session", connection_string="sqlite:///sqlite_recommend.db"
        )

        # LangGraph 기반 워크플로우 생성 (추후 RAG 기능 강화 가능)
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.memory)

        # LangChain의 메시지 히스토리 기능 연동
        self.chain = self.prompt_template | self.llm
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            lambda session_id: SQLChatMessageHistory(session_id=session_id, connection_string="sqlite:///sqlite_recommend.db"),
            input_messages_key="question",
            history_messages_key="history",
        )

    def _build_workflow(self):
        workflow = StateGraph(state_schema=self.State)
        workflow.add_node("model", self._call_model)
        workflow.add_edge(START, "model")
        return workflow

    def _call_model(self, state: State):
        # 과거 대화 기록을 불러와 현재 상태와 결합
        past_messages = self.chat_message_history.messages
        all_messages = past_messages + state["messages"]

        # 최근 사용자의 질문 추출 (없으면 기본 질문 사용)
        last_human_message = next(
            (msg.content for msg in reversed(all_messages) if isinstance(msg, HumanMessage)), 
            "안녕하세요, 어떤 추천을 원하시는지 알려주세요."
        )

        # 프롬프트 생성 (대화 히스토리, 언어, 질문, 사용자 정보 포함)
        prompt = self.prompt_template.invoke({
            "history": all_messages,
            "language": state["language"],
            "question": last_human_message,
            "user_info": state["user_info"],
        })

        # 모델 실행 및 응답 반환
        response = self.model.invoke(prompt)
        return {"messages": all_messages + [response]}

    def get_recommendation(self, question: str, user_info: dict, language: str = "ko", session_id: str = "default_session"):
        """
        사용자 질문과 개인 정보를 받아 추천 결과를 생성합니다.
        """
        input_messages = [HumanMessage(question)]
        self.chat_message_history.add_user_message(question)
        
        output = self.app.invoke(
            {"messages": input_messages, "language": language, "user_info": user_info},
            {"configurable": {"thread_id": session_id}}
        )
        
        ai_response = output["messages"][-1].content
        self.chat_message_history.add_ai_message(ai_response)
        return ai_response
