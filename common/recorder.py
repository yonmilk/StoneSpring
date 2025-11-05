import os
import cv2
import pyaudio
import threading
from datetime import datetime
from core.settings import VIDEO_CONFIG, AUDIO_CONFIG


class LiveAudioRecorder:
    def __init__(self):
        self.running = False
        self.thread = None
        self.sample_rate = AUDIO_CONFIG["sample_rate"]
        self.chunk = AUDIO_CONFIG["chunk_size"]
        self.channels = AUDIO_CONFIG["channels"]
        self.save_dir = AUDIO_CONFIG["save_dir"]
        self.session_audio = bytearray()

    def _record_loop(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

        while self.running:
            try:
                data = stream.read(self.chunk, exception_on_overflow=False)
                self.session_audio.extend(data)
            except Exception as e:
                print(f"[녹음 오류] {e}")
                break

        stream.stop_stream()
        stream.close()
        p.terminate()

    def _save_chunk(self, path, data: bytes):
        import wave
        wf = wave.open(path, "wb")
        wf.setnchannels(self.channels)
        wf.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()

    def start(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.session_audio = bytearray()
        self.running = True
        self.thread = threading.Thread(target=self._record_loop)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        final_path = None

        if self.session_audio:
            os.makedirs(self.save_dir, exist_ok=True)
            final_path = generate_filename("audio", "wav", self.save_dir)
            try:
                self._save_chunk(final_path, bytes(self.session_audio))
            except Exception as e:
                print(f"[녹음 저장 실패] {e}")
                final_path = None
        else:
            print("[녹음 실패] session_audio가 비어 있습니다.")

        self.session_audio = bytearray()
        self.thread = None
        self.running = False
        if final_path:
            return final_path
        return None  # 실패 시 None 반환

# 기존 VideoRecorder, generate_filename은 그대로 유지
class VideoRecorder:
    def __init__(self, fps=30):
        self.fps = fps
        self.running = False
        self.thread = None
        self.save_dir = VIDEO_CONFIG["save_dir"]

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._record_loop)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _record_loop(self):
        os.makedirs(self.save_dir, exist_ok=True)
        output_path = generate_filename("video", "mp4", self.save_dir)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("카메라 열기 실패")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()

def generate_filename(prefix: str, ext: str, directory: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.{ext}"
    return os.path.join(directory, filename)
