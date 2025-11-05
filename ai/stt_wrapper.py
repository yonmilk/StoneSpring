import time
from whispercpp_kit import WhisperCPP

_whisper = WhisperCPP(model_name="base")
_whisper.setup()

def transcribe_audio(file_path: str) -> str:
    """
    Whisper.cpp-ki를 이용해 음성 파일을 텍스트로 변환
    :param file_path: wav 파일 경로
    :return: 텍스트 변환 결과
    """
    try:
        result = _whisper.transcribe(file_path, language="ko")
        return result.strip()
    except Exception as e:
        print(f"[STT] 변환 오류: {e}")
        return ""

def transcribe_worker(audio_queue, callback):
    """
    실시간 음성 인식 스레드
    """
    while True:
        if audio_queue:
            file_path = audio_queue.pop(0)
            try:
                text = _whisper.transcribe(file_path, language="ko").strip()
                if text:
                    callback(text)
            except Exception as e:
                print(f"[STT] 오류: {e}")
        else:
            time.sleep(0.1)

def get_whisper():
    return _whisper
