from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from views.chat_panel import ChatPanel
from views.pet_panel import PetPanel
from views.user_status_panel import UserPanel
from views.setting_panel import SettingPanel

class MainView(QWidget):
    def __init__(self, user_id, user_name):
        super().__init__()
        layout = QVBoxLayout()
        tabs = QTabWidget()

        # 패널 인스턴스 생성
        self.user_panel = UserPanel(user_id=user_id)
        self.pet_panel = PetPanel(user_id=user_id)
        self.chat_panel = ChatPanel(user_id=user_id, user_name=user_name)
        self.setting_panel = SettingPanel(user_id=user_id)

        # 탭에 패널 추가
        tabs.addTab(self.user_panel, "나의 상태")
        tabs.addTab(self.pet_panel, "Dolbom 상태")
        tabs.addTab(self.chat_panel, "채팅")
        tabs.addTab(self.setting_panel, "설정")
        layout.addWidget(tabs)
        self.setLayout(layout)

        # 동기화 중임을 나타내는 플래그
        self._syncing = False

        # 카메라, 마이크 체크박스 동기화 연결
        self.chat_panel.cameraToggled.connect(self.sync_camera_checkbox)
        self.chat_panel.micToggled.connect(self.sync_mic_checkbox)
        self.user_panel.camera_checkbox.toggled.connect(self.sync_camera_checkbox)
        self.user_panel.mic_checkbox.toggled.connect(self.sync_mic_checkbox)

        # 감정 분석 결과 시그널 연결: ChatPanel -> UserPanel의 update_face_expression 메서드
        self.chat_panel.expressionDetected.connect(self.user_panel.update_face_expression)
        self.chat_panel.gestureDetected.connect(self.user_panel.update_gesture)

    def sync_camera_checkbox(self, checked: bool):
        if self._syncing:
            return
        self._syncing = True
        self.chat_panel.camera_checkbox.blockSignals(True)
        self.user_panel.camera_checkbox.blockSignals(True)
        self.chat_panel.camera_checkbox.setChecked(checked)
        self.user_panel.camera_checkbox.setChecked(checked)
        self.chat_panel.camera_checkbox.blockSignals(False)
        self.user_panel.camera_checkbox.blockSignals(False)
        # 실제 카메라 on/off 기능 호출 (ChatPanel에서 처리)
        self.chat_panel.toggle_camera(checked)
        self._syncing = False

    def sync_mic_checkbox(self, checked: bool):
        if self._syncing:
            return
        self._syncing = True
        self.chat_panel.mic_checkbox.blockSignals(True)
        self.user_panel.mic_checkbox.blockSignals(True)
        self.chat_panel.mic_checkbox.setChecked(checked)
        self.user_panel.mic_checkbox.setChecked(checked)
        self.chat_panel.mic_checkbox.blockSignals(False)
        self.user_panel.mic_checkbox.blockSignals(False)
        # 실제 마이크 on/off 기능 호출 (ChatPanel에서 처리)
        self.chat_panel.toggle_mic(checked)
        self._syncing = False
