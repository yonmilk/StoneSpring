import os
from openai import OpenAI

API_KEY = os.getenv("OPENAI_API_KEY") 
GPT_MODEL_NAME="gpt-4o-mini"

client = OpenAI(api_key=API_KEY)

INTRO_PROMPT = (
    "너는 감정 케어를 도와주는 대화형 AI 'Dolbom'야.\n"
    "상대방의 감정에 공감하고, 따뜻하고 다정한 말투로 위로와 응원을 건네는 역할을 해.\n\n"
    "대화를 시작하기 전, 다음의 설정 정보를 참고해야 해:\n"
    "- [말투]: 문장 끝 어미에 반영되는 말투 스타일이야. 예: 존댓말, 반말 등\n"
    "- [성격]: 응답의 어조와 분위기에 영향을 주는 Dolbom의 성격이야. 예: 내향적, 외향적\n"
    "- [참고 설정]: 사용자가 전달한 추가 정보로, 대화 중 필요할 수 있는 중요한 참고사항이야\n"
    "- [훈련 명령 목록]: 사용자가 Dolbom에게 훈련시킨 명령어와 대응 행동이야. 대화 중 관련된 문맥이 나타나면 활용해\n\n"
    "반드시 한국어로만 응답하고, 설정된 말투와 성격을 반영해서 대화해.\n"
    "너무 길거나 장황하지 않게, 핵심적이고 감성적으로 응답하는 것이 좋아."
)

def build_input_messages(user_message: str, character: dict, trainings: list,
                         chat_history: list = None, max_trainings: int = 5) -> list:
    messages = [
        {"role": "system", "content": INTRO_PROMPT},
        {"role": "system", "content": f"[말투] → {character.get('speech', '존댓말')}"},
        {"role": "system", "content": f"[성격] → {character.get('character', '내향적')}"},
        {"role": "system", "content": f"[참고 설정] → {character.get('resSetting', '')}"}
    ]

    if trainings:
        limited = trainings[:max_trainings]
        training_lines = "\n".join(
            f"- {t.get('trainingText', '')} → {t.get('recognizedGesture', '')}"
            for t in limited
        )
        messages.append({"role": "system", "content": f"[훈련 명령 목록]\n{training_lines}"})

    if chat_history:
        messages.extend(chat_history)

    # 마지막 사용자 입력 추가
    messages.append({"role": "user", "content": user_message})

    return messages

def moderate_text(text: str) -> bool:
    """
    주어진 텍스트가 OpenAI Moderation에 의해 차단되는지 확인.
    부적절하면 True(차단 필요 -> 응답 생성 X)
    적절하면 False(정상 처리 -> 응답 생성 진행)
    """
    moderation = client.moderations.create(input=text)
    flagged = any(result.flagged for result in moderation.results)
    return flagged

def generate_reply(user_message: str,
                   character: dict,
                   trainings: list,
                   chat_history: list = None,
                   previous_response_id: str = None,
                   model: str = "gpt-4o-mini",
                   stream: bool = True):
    """
    사용자 입력, 캐릭터 설정, 훈련 목록, 히스토리를 바탕으로 응답 생성 (stream/non-stream 지원)

    Returns:
        - response: stream generator or 단일 응답
        - response_id: 응답 ID (non-stream 시 반환됨)
    """
    if moderate_text(user_message):
        return "⚠️ 입력 내용에 부적절한 요소가 포함되어 있어요. ⚠️", None

    input_messages = build_input_messages(
        user_message=user_message,
        character=character,
        trainings=trainings,
        chat_history=chat_history
    )

    tools = [{"type": "web_search_preview"}] if "웹 검색" in user_message else None

    response = client.responses.create(
        model=model,
        input=input_messages,
        tools=tools,
        temperature=0.7,
        top_p=0.9,
        stream=stream,
        **({ "previous_response_id": previous_response_id } if not stream and previous_response_id else {})
    )

    if stream:
        def stream_chunks():
            for event in response:
                event_type = getattr(event, "type", "")
                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        if isinstance(delta, str):
                            text = delta
                        elif hasattr(delta, "text"):
                            text = getattr(delta, "text", "")
                        else:
                            text = str(delta)
                        if text:
                            yield text
                elif event_type == "response.error":
                    error_msg = getattr(event, "error", "")
                    message = str(error_msg) if error_msg else "OpenAI 응답 처리 중 오류가 발생했습니다."
                    yield message
                    break
                elif event_type == "response.completed":
                    break

        return stream_chunks(), None

    return response.output_text.strip(), response.id
