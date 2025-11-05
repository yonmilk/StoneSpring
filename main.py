import sys
import threading
import os
import cv2
import PyQt5
from PyQt5.QtWidgets import QApplication
from views.main_window import MainWindow
from network.tcp_server import start_server

qt_plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), 'Qt', 'plugins')
os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', qt_plugin_path)

if __name__ == '__main__':
    threading.Thread(target=start_server, daemon=True).start()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
