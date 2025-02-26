# data_processor.py
def preprocess_input_data(data: dict) -> dict:
    """
    server.py에서 전달된 원본 데이터를 정제하여 필요한 정보를 추출.
    1. 필수 데이터가 없을 경우 기본값 설정
    2. MBTI는 대문자로 변환
    """
    return {
        "user_info": data.get("user_info", {}),
        "gpt_mbti": data.get("gpt_mbti", {}).get("MBTI", "").upper(),  # ✅ MBTI 대문자 변환
        "fortune": data.get("fortune", {}),
        "vs_data": data.get("vs_data", {}),
        "question": data.get("question", "추천을 받고 싶어요.")  # ✅ 기본 질문 설정
    }


def process_prompt_data(clean_data: dict) -> str:
    """
    정제된 데이터를 받아 GPT 프롬프트를 생성.
    """
    user_info = clean_data["user_info"]
    gpt_mbti = clean_data["gpt_mbti"]
    fortune = clean_data["fortune"]
    vs_data = clean_data["vs_data"]
    question = clean_data["question"]
    
    # MBTI 역할 지시 (간단한 매핑)
    mbti_role = get_mbti_role(gpt_mbti)
    
    # vs 데이터(취향 정보) 요약: 예를 들어 "커피_vs_차"에서 앞쪽 단어만 추출
    taste_summary = ", ".join(
        [f"{key.split('_vs_')[0]}: {value}" for key, value in vs_data.items()]
    )
    
    # 프롬프트 구성 (여러 정보를 균형 있게 포함)
    prompt = (
        f"**사용자 정보**: {user_info.get('birth', '미제공')}, {user_info.get('gender', '미제공')}\n"

        f"**GPT 역할**: {mbti_role}\n"

        f"**오늘의 운세**: {fortune.get('daily', '운세 정보 없음')}\n"
        f"**사주 분석**: {fortune.get('saju', '사주 정보 없음')}\n"

        f"**사용자 취향**: {taste_summary if taste_summary else '취향 정보 없음'}\n"

        f"**질문**: {question}\n\n"

        "위 정보를 바탕으로, 운세·사주 및 사용자의 취향을 종합하여 최적의 추천을 생성하세요.\n"
        "답변은 MBTI 특성을 반영한 자연스러운 말투로 제공해주세요.\n"
        "추천은 단순한 취향 선택을 따르기보다, 운세와 상황을 고려해 유연하고 창의적인 옵션을 포함해주세요."
    )

    return prompt


def get_mbti_role(gpt_mbti: str) -> str:
    """
    MBTI에 따른 역할 지시를 반환합니다.
    예) INFJ인 경우, "섬세하고 공감 능력이 뛰어나며, 창의적이고 예술적인 감각을 지님" 등
    """

    # MBTI 매핑
    mbti_mapping = {
        "INFJ": "섬세하고 공감 능력이 뛰어나며, 창의적이고 예술적인 감각을 지님.",
        "ENFP": "열정적이고 창의적이며, 새로운 아이디어를 탐구하는 것을 좋아함.",
        "INTJ": "논리적이고 전략적이며, 미래를 내다보는 분석력을 보유함.",
        "ENTP": "도전적이고 개방적이며, 토론을 즐기고 창의적인 해결책을 찾음.",
        "ISTJ": "철저하고 신뢰할 수 있으며, 체계적이고 책임감이 강함.",
        "ESTJ": "현실적이고 지도력이 뛰어나며, 조직적이고 목표 지향적임.",
        "ISFJ": "따뜻하고 헌신적이며, 세심하게 타인을 배려하는 성향이 강함.",
        "ESFJ": "사교적이고 친절하며, 사람들과의 관계를 중요하게 생각함.",
        "ISTP": "유연하고 논리적이며, 현실적인 문제 해결 능력이 뛰어남.",
        "ESTP": "활발하고 즉흥적이며, 현실 감각이 뛰어나고 도전 정신이 강함.",
        "ISFP": "예술적이고 감성적이며, 조용하지만 내면이 깊고 창의적임.",
        "ESFP": "활발하고 사교적이며, 새로운 경험을 즐기고 분위기를 띄우는 역할을 함.",
        "INTP": "논리적이고 분석적이며, 개념적인 사고와 탐구를 즐김.",
        "ENTJ": "결단력 있고 지도력이 뛰어나며, 목표를 달성하는 데 능숙함.",
        "ENFJ": "사람을 잘 이해하고 이끄는 능력이 뛰어나며, 타인의 성장을 돕는 것을 좋아함.",
        "INFP": "이상적이고 감성적이며, 창의적이면서도 깊은 내면을 가진 성향."
    }

    # 매핑된 MBTI 설명 반환 (없으면 기본값 반환)
    return mbti_mapping.get(gpt_mbti, "균형 잡힌 성향을 지닌 사용자")
