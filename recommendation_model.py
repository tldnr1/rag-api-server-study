# import sqlite3
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage #, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_openai import ChatOpenAI
#from langchain_core.runnables.history import RunnableWithMessageHistory

# 데이터 처리 함수 로드
from data_processor import process_prompt_data

# API KEY 자동 로드
load_dotenv()


class RecommendationModel:
    """
    GPT-4o 기반 추천 시스템.
    """

    def __init__(self, model_name="gpt-4o-mini", model_provider="openai"):
        """
        모델 및 대화 히스토리 초기화.
        """
        # LangChain 모델 초기화
        self.model = init_chat_model(model_name, model_provider=model_provider)
        self.llm = ChatOpenAI(model_name=model_name)

        # 프롬프트 템플릿 설정
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "{context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        # SQLite 기반 대화 히스토리 저장 설정
        self.db_connection = "sqlite:///sqlite_recommend.db"

    def _get_chat_history(self, session_id: str):
        """
        특정 세션의 대화 히스토리를 가져옴.
        """
        chat_history = SQLChatMessageHistory(session_id=session_id, connection_string=self.db_connection)
        return chat_history.messages

    def _get_last_user_message(self, messages):
        """
        사용자 메시지 중 가장 최근 메시지를 추출.
        """
        return next(
            (msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)),
            "안녕하세요, 어떤 추천을 원하시는지 알려주세요."
        )

    def get_recommendation(self, clean_data: dict, session_id: str = "default_session"):
        """
        정제된 데이터를 기반으로 GPT 프롬프트를 생성하고 실행.
        """
        # 1. 대화 히스토리 로드
        past_messages = self._get_chat_history(session_id)
        input_messages = [HumanMessage(clean_data["question"])]
        all_messages = past_messages + input_messages

        # 2. 최근 질문 추출
        last_human_message = self._get_last_user_message(all_messages)

        # 3. 프롬프트 생성
        context = process_prompt_data(clean_data)

        # 4. 프롬프트 템플릿 적용
        prompt = self.prompt_template.invoke({
            "context": context,
            "history": all_messages,
            "question": last_human_message
        })

        # 5. GPT 실행
        response = self.model.invoke(prompt)

        # 6. 대화 히스토리 저장
        chat_history = SQLChatMessageHistory(session_id=session_id, connection_string=self.db_connection)
        chat_history.add_user_message(last_human_message)
        chat_history.add_ai_message(response.content)

        # 7. 토큰 사용량 분석
        show_token_result(str(prompt), response.content)

        return response.content


''' 토큰 사용량 분석 함수 '''
import tiktoken

def get_token_count(text: str, model_name: str = "gpt-4o-mini") -> int:
    """
    텍스트의 토큰 개수를 반환하는 함수.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def show_token_result(prompt: str, response_text: str, model_name: str = "gpt-4o-mini"):
    """
    프롬프트와 응답의 토큰 사용량을 분석하여 출력.
    """
    input_token_count = get_token_count(prompt, model_name)
    output_token_count = get_token_count(response_text, model_name)

    input_rate = 0.150 / 1_000_000  # $ per token for input tokens
    output_rate = 0.600 / 1_000_000  # $ per token for output tokens

    iteration_cost = (input_token_count * input_rate) + (output_token_count * output_rate)
    estimated_cost_100 = iteration_cost * 100

    print(f"[DEBUG] Input token count: {input_token_count}")
    print(f"[DEBUG] Output token count: {output_token_count}")
    print(f"[DEBUG] Cost for this iteration: ${iteration_cost:.8f}")
    print(f"[DEBUG] Estimated cost for 100 iterations: ${estimated_cost_100:.8f}")
