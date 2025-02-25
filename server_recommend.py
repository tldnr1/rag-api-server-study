# server.py
from fastapi import FastAPI, Query, Body
from fastapi.responses import JSONResponse
import sql_chat_model
from recommendation_model import RecommendationModel

app = FastAPI()

# 기존 번역 모델 인스턴스 (대화 기록 저장 등 동일)
translation_model = sql_chat_model.TranslationModel()

# 새 추천 모델 인스턴스 (개인 정보 반영 추천)
recommendation_model = RecommendationModel()

@app.get("/translates")
def translate(
    text: str = Query(..., description="번역할 텍스트"),
    language: str = Query(default="ko", description="대답할 언어"),
    session_id: str = Query(default="default_thread", description="대화 세션 ID")
):
    response = translation_model.translate(text, language, session_id)
    return {"content": response}

@app.post("/recommend")
def recommend(
    data: dict = Body(..., example={
        "question": "내일 데이트 가는데 어떤 옷이 좋을까?",
        "user_info": {
            "birth": "1990-01-01",
            "gender": "여성",
            "MBTI": "INFJ"
            # 추후 운세 정보 등 추가 가능
        },
        "language": "ko",
        "session_id": "user123"
    })
):
    """
    사용자의 질문과 개인 정보를 받아 개인 맞춤형 추천 결과를 반환합니다.
    - **question**: 사용자의 질문 (예: 데이트 옷 추천)
    - **user_info**: 사용자 정보 (생년월일, 성별, MBTI 등)
    - **language**: 응답 언어 (기본값: ko)
    - **session_id**: 대화 세션 ID (기본값: default_session)
    """
    question = data.get("question")
    user_info = data.get("user_info")
    language = data.get("language", "ko")
    session_id = data.get("session_id", "default_session")

    if not question or not user_info:
        return JSONResponse(
            status_code=400,
            content={"error": "Both 'question' and 'user_info' fields are required."}
        )

    recommendation = recommendation_model.get_recommendation(question, user_info, language, session_id)
    return {"recommendation": recommendation}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_recommend:app", host="0.0.0.0", port=8000, reload=True)
