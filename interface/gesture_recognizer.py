import json
import os
import threading
from collections import deque
from queue import Queue, Empty

import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from tensorflow.keras.models import load_model


class GestureRecognizer(QObject):
    """
    공유 카메라 프레임을 이용해 제스처를 실시간으로 인식하는 헬퍼 클래스.
    CameraManager로부터 프레임 콜백을 등록받아 사용하며,
    인식된 제스처는 gestureRecognized 시그널로 GUI 쓰레드에 전달됩니다.
    """

    gestureRecognized = pyqtSignal(str, float)
    statusChanged = pyqtSignal(str)

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        min_confidence: float = 0.8,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.model_path = model_path
        self.labels_path = labels_path
        self.min_confidence = min_confidence

        self.seq_length = 30
        self.seq = deque(maxlen=self.seq_length)
        self.action_seq = deque(maxlen=3)
        self.frame_queue: Queue[np.ndarray | None] = Queue(maxsize=2)

        self.running = False
        self.worker: threading.Thread | None = None
        self.model = None
        self.actions = None
        self.hands = None

    def start(self):
        if self.running:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"제스처 모델 파일을 찾을 수 없습니다: {self.model_path}")
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"제스처 라벨 파일을 찾을 수 없습니다: {self.labels_path}")

        self.model = load_model(self.model_path)
        with open(self.labels_path, "r", encoding="utf-8") as f:
            self.actions = json.load(f)

        hands_module = mp.solutions.hands
        self.hands = hands_module.Hands(
            max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        self.running = True
        self.worker = threading.Thread(target=self._process_loop, daemon=True)
        self.worker.start()
        self.statusChanged.emit("제스처 인식 시작")

    def stop(self):
        if not self.running:
            return

        self.running = False
        self.frame_queue.put(None)
        if self.worker:
            self.worker.join(timeout=1.0)
        self.worker = None

        if self.hands:
            self.hands.close()
        self.hands = None

        self.seq.clear()
        self.action_seq.clear()
        self.model = None
        self.actions = None
        self.statusChanged.emit("제스처 인식 중지")

    def submit_frame(self, frame):
        if not self.running or frame is None:
            return
        try:
            self.frame_queue.put_nowait(frame.copy())
        except Exception:
            # 큐가 가득 찬 경우 가장 최근 프레임만 유지
            try:
                _ = self.frame_queue.get_nowait()
            except Empty:
                pass
            try:
                self.frame_queue.put_nowait(frame.copy())
            except Exception:
                pass

    def _process_loop(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.2)
            except Empty:
                continue

            if frame is None:
                continue

            try:
                self._process_frame(frame)
            except Exception as exc:
                self.statusChanged.emit(f"제스처 처리 오류: {exc}")

    def _process_frame(self, frame):
        if self.hands is None or self.model is None or self.actions is None:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)

        if not result.multi_hand_landmarks:
            self.action_seq.clear()
            return

        for hand_landmarks in result.multi_hand_landmarks:
            joint = np.zeros((21, 4), dtype=np.float32)
            for idx, landmark in enumerate(hand_landmarks.landmark):
                joint[idx] = [landmark.x, landmark.y, landmark.z, landmark.visibility]

            v1 = joint[
                [
                    0,
                    1,
                    2,
                    3,
                    0,
                    5,
                    6,
                    7,
                    0,
                    9,
                    10,
                    11,
                    0,
                    13,
                    14,
                    15,
                    0,
                    17,
                    18,
                    19,
                ],
                :3,
            ]
            v2 = joint[
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                ],
                :3,
            ]
            v = v2 - v1
            norms = np.linalg.norm(v, axis=1)
            norms[norms == 0] = 1e-6
            v /= norms[:, np.newaxis]

            angle = np.arccos(
                np.clip(
                    np.einsum(
                        "nt,nt->n",
                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :],
                    ),
                    -1.0,
                    1.0,
                )
            )
            angle = np.degrees(angle)

            feature_vector = np.concatenate([joint.flatten(), angle])
            self.seq.append(feature_vector)

            if len(self.seq) < self.seq_length:
                continue

            input_data = np.expand_dims(np.array(self.seq, dtype=np.float32), axis=0)
            y_pred = self.model.predict(input_data, verbose=0).squeeze()
            if y_pred.ndim == 0:
                continue

            idx_pred = int(np.argmax(y_pred))
            confidence = float(y_pred[idx_pred])
            if confidence < self.min_confidence:
                continue

            predicted_action = self.actions[idx_pred]
            self.action_seq.append(predicted_action)

            if len(self.action_seq) == self.action_seq.maxlen and len(set(self.action_seq)) == 1:
                self.action_seq.clear()
                self.gestureRecognized.emit(predicted_action, confidence)
