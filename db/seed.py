from db.models import *
from db.query import *
from datetime import datetime, timedelta

# 기본 사용자 생성
user = User(
    user_id=None,
    name="홍길동",
    nickname="길동이",
    email="test@example.com",
    password="hashed_password"
)
user_id = insert_user(user)
print(f"[더미 생성] 사용자 ID: {user_id}")

# 펫 성격 설정
character = PetCharacterSetting(
    character_id=None,
    user_id=user_id,
    speech="존댓말",
    character_style="내향적",
    res_setting="땅콩 알레르기 있음"
)
insert_character_setting(character)
print("[더미 생성] 펫 성격 설정 완료")

# 훈련 명령 설정
training = PetTrainingSetting(
    training_setting_id=None,
    user_id=user_id,
    training_text="도와줘",
    keyword_text="도움, 위험",
    gesture_video_path="/gesture/help.mp4",
    gesture_recognition_id=1,
    recognized_gesture="손 흔들기"
)
insert_training_setting(training)
print("[더미 생성] 훈련 명령 설정 완료")

# 사용자 설정
setting = UserSetting(
    setting_id=None,
    user_id=user_id,
    font_size=16
)
set_user_setting(setting)
print("[더미 생성] 사용자 설정 완료")

# 펫 이모티콘 추가
emoticon = PetEmoticon(
    e_id=None,
    emoticon="ʕ◍·̀Ⱉ·́◍ʔ",
    text="기뻐요"
)
insert_pet_emoticon(emoticon)
print("[더미 생성] 펫 이모티콘 추가 완료")

# 채팅 메시지 삽입
now = datetime.now()
video_end = now + timedelta(seconds=2)
voice_end = now + timedelta(seconds=2)

chat = Chat(
    chat_id=None,
    user_id=user_id,
    message_id=101,
    message="기분이 좀 이상해...",
    timestamp=now.strftime('%Y-%m-%d %H:%M:%S'),
    video_id=1,
    video_path="/video/101.mp4",
    video_start_timestamp=now.strftime('%Y-%m-%d %H:%M:%S'),
    video_end_timestamp=video_end.strftime('%Y-%m-%d %H:%M:%S'),
    voice_id=2,
    voice_path="/voice/201.wav",
    voice_start_timestamp=now.strftime('%Y-%m-%d %H:%M:%S'),
    voice_end_timestamp=voice_end.strftime('%Y-%m-%d %H:%M:%S'),
    e_id=None,
    pet_emotion="걱정해요",
    reply_message="괜찮아, 곧 나아질 거야"
)
chat_id = insert_chat(chat)
print(f"[더미 생성] 채팅 삽입 완료 (chat_id: {chat_id})")

# 감정 분석 삽입
emotion = UserEmotionAnalysis(
    emotion_analysis_id=None,
    user_id=user_id,
    chat_id=chat_id,
    face_emotion="sad",
    voice_emotion="low",
    text_emotion="worried",
    summary="걱정과 불안이 감지됨",
    time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
)
insert_user_emotion_analysis(emotion)
print("[더미 생성] 감정 분석 데이터 삽입 완료")

# 로그 기록
log = Log(
    log_id=None,
    user_id=user_id,
    log_type="emotion_analysis",
    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    detail="감정 분석 완료",
    location="desktop",
    device_info="Windows 11",
    error_code=None,
    ip_address="127.0.0.1"
)
insert_log(log)
print("[더미 생성] 로그 기록 완료")
