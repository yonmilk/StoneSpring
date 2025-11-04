from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QFont, QImage, QPixmap
import cv2
from datetime import datetime
from interface.camera_manager import CameraManager
from interface.emotion import analyze_emotion as analyze_webcam_emotion

class UserPanel(QWidget):
    cameraToggled = pyqtSignal(bool)
    micToggled = pyqtSignal(bool)
    gestureToggled = pyqtSignal(bool)

    def __init__(self, user_id=None):
        super().__init__()
        self.user_id = user_id
        self.camera_on = False
        self.gesture_on = False
        self.last_emotion_update = 0
        self.camera_manager = None
        self.current_expression_text = ""  # 카메라 오버레이용 감정 문자열
        self.current_gesture_text = ""

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignTop)

        toggle_layout = QHBoxLayout()
        self.camera_checkbox = QCheckBox("카메라 ON")
        self.mic_checkbox = QCheckBox("마이크 ON")
        self.gesture_checkbox = QCheckBox("제스처 ON")
        toggle_layout.addWidget(self.camera_checkbox)
        toggle_layout.addWidget(self.mic_checkbox)
        toggle_layout.addWidget(self.gesture_checkbox)
        toggle_layout.addStretch()

        self.camera_checkbox.toggled.connect(self.emit_camera_toggled)
        self.mic_checkbox.toggled.connect(self.emit_mic_toggled)
        self.gesture_checkbox.toggled.connect(self.emit_gesture_toggled)

        info_layout = QVBoxLayout()
        info_layout.setAlignment(Qt.AlignTop)
        self.face_label = QLabel("● 표정 정보")
        self.face_emotion_label = QLabel("감정: 없음")
        info_layout.addWidget(self.face_label)
        info_layout.addWidget(self.face_emotion_label)

        self.voice_label = QLabel("● 목소리 정보")
        self.voice_emotion_label = QLabel("목소리 정보 없음")
        info_layout.addWidget(self.voice_label)
        info_layout.addWidget(self.voice_emotion_label)

        self.gesture_label = QLabel("● 제스처 정보")
        self.gesture_status_label = QLabel("제스처 정보 없음")
        info_layout.addWidget(self.gesture_label)
        info_layout.addWidget(self.gesture_status_label)

        # 카메라 출력 영역
        self.camera_view_label = QLabel("카메라 화면 출력")
        self.camera_view_label.setStyleSheet("background-color: black; color: white;")
        self.camera_view_label.setAlignment(Qt.AlignCenter)
        self.camera_view_label.setFixedSize(500, 375)

        main_layout.addLayout(toggle_layout)
        main_layout.addLayout(info_layout)
        main_layout.addWidget(self.camera_view_label, alignment=Qt.AlignHCenter)

        self.setLayout(main_layout)

    def emit_camera_toggled(self, checked: bool):
        self.camera_on = checked
        self.update_camera_view_visibility()
        self.cameraToggled.emit(checked)
        if checked:
            self.start_camera()
        else:
            self.stop_camera()

    def emit_mic_toggled(self, checked: bool):
        self.micToggled.emit(checked)

    def emit_gesture_toggled(self, checked: bool):
        self.gesture_on = checked
        self.update_camera_view_visibility()
        self.gestureToggled.emit(checked)
        if not checked:
            self.update_gesture("", 0.0)

    def update_camera_view_visibility(self):
        if self.camera_on or self.gesture_on:
            self.camera_view_label.show()
        else:
            self.camera_view_label.hide()

    def start_camera(self):
        try:
            self.camera_manager = CameraManager.instance()
        except RuntimeError as e:
            print("카메라 시작 오류:", e)
            self.camera_checkbox.setChecked(False)
            return
        self.camera_manager.add_frame_callback(self.update_camera_frame)
        self.camera_manager.start()

    def stop_camera(self):
        if self.camera_manager is not None:
            self.camera_manager.remove_frame_callback(self.update_camera_frame)
        self.camera_view_label.clear()

    def update_camera_frame(self, frame):
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image).scaled(
                self.camera_view_label.width(), self.camera_view_label.height(), Qt.KeepAspectRatio
            )

            overlay_lines = []
            if self.current_expression_text:
                overlay_lines.append(self.current_expression_text)
            if self.current_gesture_text:
                overlay_lines.append(self.current_gesture_text)

            if overlay_lines:
                painter = QPainter(pixmap)
                painter.setPen(QColor(0, 255, 0))  # 초록색 텍스트
                font = QFont("Arial", 16)
                font.setBold(True)
                painter.setFont(font)
                for idx, line in enumerate(overlay_lines):
                    painter.drawText(10, 30 + idx * 30, line)
                painter.end()

            self.camera_view_label.setPixmap(pixmap)

            # 1초마다 감정 분석 업데이트
            current_time = datetime.now().timestamp()
            if current_time - self.last_emotion_update >= 1:
                summary = analyze_webcam_emotion(frame)
                self.face_emotion_label.setText(summary)
                self.current_expression_text = summary
                self.last_emotion_update = current_time

    def update_face_expression(self, expression: str):
        self.face_emotion_label.setText(expression)
        self.current_expression_text = expression

    def update_gesture(self, gesture: str, confidence: float):
        if gesture:
            text = f"제스처: {gesture} ({confidence * 100:.1f}%)"
            self.gesture_status_label.setText(text)
            self.current_gesture_text = text
        else:
            self.gesture_status_label.setText("제스처 정보 없음")
            self.current_gesture_text = ""


