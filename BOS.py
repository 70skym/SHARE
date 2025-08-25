import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QSlider, QSpinBox, QGridLayout, QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

def local_phase_correlation_map(img1, img2, block_size=32, step=8):
    h, w = img1.shape
    out_h = (h - block_size) // step + 1
    out_w = (w - block_size) // step + 1
    result = np.zeros((out_h, out_w), dtype=np.float32)

    for i, y in enumerate(range(0, h - block_size + 1, step)):
        for j, x in enumerate(range(0, w - block_size + 1, step)):
            patch1 = img1[y:y+block_size, x:x+block_size].astype(np.float32)
            patch2 = img2[y:y+block_size, x:x+block_size].astype(np.float32)
            F1 = np.fft.fft2(patch1)
            F2 = np.fft.fft2(patch2)
            R = F1 * np.conj(F2)
            R /= np.abs(R) + 1e-8
            corr = np.abs(np.fft.ifft2(R))
            result[i, j] = np.max(corr)

    # 正規化と反転（相関低→変位大）
    result = 1.0 - result / np.max(result)

    # 出力サイズに合わせて補間（双線形で画質保つ）
    result_resized = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
    return (255 * result_resized).astype(np.uint8)

class FullscreenBackground(QWidget):
    def __init__(self, width, height, dot_size):
        super().__init__()
        self.setWindowTitle("Background Pattern")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.showFullScreen()
        self.bg_image = self.create_random_dot_pattern(width, height, dot_size)
        self.label = QLabel()
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.update_image(self.bg_image)

    def create_random_dot_pattern(self, w, h, dot_size):
        img = np.ones((h, w), dtype=np.uint8) * 255
        num_dots = w * h // (dot_size ** 2 * 2)
        for _ in range(num_dots):
            x = np.random.randint(0, w - dot_size)
            y = np.random.randint(0, h - dot_size)
            img[y:y + dot_size, x:x + dot_size] = 0
        return img

    def update_image(self, img):
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

class BOSDisplayWindow(QWidget):
    def __init__(self, title="Reconstruction Display"):
        super().__init__()
        self.setWindowTitle(title)
        self.label = QLabel()
        self.label.setFixedSize(800, 600)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    def update_image(self, img_bgr):
        h, w, ch = img_bgr.shape
        qimg = QImage(img_bgr.data, w, h, ch * w, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

class BOSApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time BOS GUI")
        self.ref_image = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.init_ui()
        self.display_window = None
        self.fullscreen_bg = None

    def init_ui(self):
        layout = QGridLayout()
        layout.addWidget(QLabel("Real-time BOS GUI"), 0, 0, 1, 3)
        layout.addWidget(QLabel("Background Width:"), 1, 0)
        self.width_spin = QSpinBox(); self.width_spin.setRange(100, 3840); self.width_spin.setValue(3440)
        layout.addWidget(self.width_spin, 1, 1)
        layout.addWidget(QLabel("Background Height:"), 2, 0)
        self.height_spin = QSpinBox(); self.height_spin.setRange(100, 2160); self.height_spin.setValue(1440)
        layout.addWidget(self.height_spin, 2, 1)

        layout.addWidget(QLabel("Dot Size:"), 3, 0)
        self.dot_slider = QSlider(Qt.Horizontal); self.dot_slider.setRange(1, 100); self.dot_slider.setValue(1)
        self.dot_spin = QSpinBox(); self.dot_spin.setRange(1, 100); self.dot_spin.setValue(1)
        layout.addWidget(self.dot_slider, 3, 1)
        layout.addWidget(self.dot_spin, 3, 2)
        self.dot_slider.valueChanged.connect(self.dot_spin.setValue)
        self.dot_spin.valueChanged.connect(self.dot_slider.setValue)

        self.bg_button = QPushButton("Generate Background"); self.bg_button.clicked.connect(self.show_fullscreen_background)
        self.capture_button = QPushButton("Capture Reference Image"); self.capture_button.clicked.connect(self.capture_reference)
        self.start_button = QPushButton("Start"); self.start_button.clicked.connect(self.start_reconstruction)
        self.stop_button = QPushButton("Stop"); self.stop_button.clicked.connect(self.stop_reconstruction)
        self.quit_button = QPushButton("Quit"); self.quit_button.clicked.connect(self.quit_app)

        layout.addWidget(self.bg_button, 4, 0)
        layout.addWidget(self.capture_button, 4, 1)
        layout.addWidget(self.start_button, 5, 0)
        layout.addWidget(self.stop_button, 5, 1)

        layout.addWidget(QLabel("CLAHE ClipLimit:"), 6, 0)
        self.clahe_slider = QSlider(Qt.Horizontal); self.clahe_slider.setRange(1, 50); self.clahe_slider.setValue(20)
        layout.addWidget(self.clahe_slider, 6, 1, 1, 2)

        layout.addWidget(QLabel("Colormap:"), 7, 0)
        self.colormap_combo = QComboBox(); self.colormap_combo.addItems(["JET", "TURBO", "BONE"])
        layout.addWidget(self.colormap_combo, 7, 1)

        layout.addWidget(QLabel("Method:"), 8, 0)
        self.method_combo = QComboBox(); self.method_combo.addItems(["DIFF", "POC", "NCC", "LOCALPOC"])
        layout.addWidget(self.method_combo, 8, 1)

        layout.addWidget(self.quit_button, 9, 0)
        self.setLayout(layout)
        self.setMinimumWidth(450)

    def quit_app(self):
        self.stop_reconstruction()
        QApplication.quit()

    def show_fullscreen_background(self):
        w, h, d = self.width_spin.value(), self.height_spin.value(), self.dot_spin.value()
        self.fullscreen_bg = FullscreenBackground(w, h, d)
        self.fullscreen_bg.show()

    def capture_reference(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened(): return
        cv2.waitKey(3000)
        ret, frame = self.cap.read()
        if ret: self.ref_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def start_reconstruction(self):
        if self.ref_image is None: return
        if self.cap is None or not self.cap.isOpened(): self.cap = cv2.VideoCapture(0)
        if self.display_window is None: self.display_window = BOSDisplayWindow("BOS Visualization")
        self.display_window.show()
        self.update_frame()
        self.timer.start(30)

    def stop_reconstruction(self):
        self.timer.stop()
        if self.cap: self.cap.release()

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened(): return
        ret, frame = self.cap.read()
        if not ret: return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        clip_limit = self.clahe_slider.value() / 10.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        ref_eq = clahe.apply(self.ref_image)
        gray_eq = clahe.apply(gray)

        method = self.method_combo.currentText()
        if method == "DIFF":
            diff = cv2.absdiff(ref_eq, gray_eq)
            diff_color = cv2.applyColorMap(diff, self.get_colormap())

        elif method == "POC":
            F1 = np.fft.fft2(ref_eq.astype(np.float32))
            F2 = np.fft.fft2(gray_eq.astype(np.float32))
            R = F1 * np.conj(F2)
            R /= np.abs(R) + 1e-8
            corr = np.fft.ifft2(R)
            corr = np.fft.fftshift(corr)
            poc_map = np.abs(corr)
            poc_norm = cv2.normalize(poc_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            diff_color = cv2.applyColorMap(poc_norm, self.get_colormap())

        elif method == "NCC":
            result = cv2.matchTemplate(gray_eq, ref_eq, cv2.TM_CCORR_NORMED)
            result_norm = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            diff_color = cv2.applyColorMap(result_norm, self.get_colormap())

        elif method == "LOCALPOC":
            pocmap = local_phase_correlation_map(ref_eq, gray_eq, block_size=32, step=16)
            diff_color = cv2.applyColorMap(pocmap, self.get_colormap())

        else:
            diff_color = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

        self.display_window.update_image(diff_color)

    def get_colormap(self):
        name = self.colormap_combo.currentText()
        return {"JET": cv2.COLORMAP_JET, "TURBO": cv2.COLORMAP_TURBO, "BONE": cv2.COLORMAP_BONE}.get(name, cv2.COLORMAP_JET)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = BOSApp()
    win.show()
    sys.exit(app.exec_())