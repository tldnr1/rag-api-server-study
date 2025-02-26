# server.py
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from recommendation_model import RecommendationModel

# 다른 파일에서 로드
from data_processor import preprocess_input_data

app = FastAPI()

# CORS 설정 (필요한 도메인만 지정 가능)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 단계에서는 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 인스턴스 생성
gpt_model = RecommendationModel(model_name="gpt-4o-mini")

@app.post("/fortune")
def get_fortune(data: dict = Body(..., example={
    "question": "내일 데이트 가는데 어떤 옷을 입어야 할까요?",
    "user_info": {"birth": "1990-01-01", "gender": "여성"},
    "gpt_mbti": {"MBTI": "INFJ"},
    "fortune": {"daily": "대인 관계 운 상승, 재물운 평범", "saju": "정서적 안정감과 창의력이 돋보이는 하루"},
    "vs_data": {"커피_vs_차": "커피", "산_vs_바다": "바다", "원피스_vs_블라우스_스커트": "원피스"}
})):
    # 최초 데이터 정제해서
    clean_data = preprocess_input_data(data)
    # 모델에 프롬프트 전달하고 응답 받고
    response_text = gpt_model.get_recommendation(clean_data, session_id="default_session")
    # 반환 데이터를 한번 더 정제해서 전달 (fe, be와 의논 필요)
    return {"response": response_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)