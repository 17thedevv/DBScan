import sys
import numpy as np
import pydicom
import cv2
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout
from PyQt6.QtGui import QPixmap, QImage, QIcon
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

class App:
    def __init__(self):
        self.window = QWidget()
        self.window.setWindowIcon(QIcon("assets\\apple.ico"))
        self.window.setWindowTitle("Ứng dụng DBScan by 17thedev")
        self.window.setGeometry(100, 100, 888, 666)

        # Giao diện
        self.label = QLabel("Chọn ảnh DICOM", self.window)
        self.label.setStyleSheet("font-size: 16px;")
        self.btn_open = QPushButton("Mở Ảnh", self.window)
        self.btn_open.setFixedSize(100, 40)
        self.btn_preprocess = QPushButton("Tiền xử lý", self.window)
        self.btn_preprocess.setFixedSize(100, 40)
        self.btn_scan = QPushButton("DBScan", self.window)
        self.btn_scan.setFixedSize(100, 40)

        self.btnLayout = QHBoxLayout()
        self.btnLayout.addWidget(self.btn_open)
        self.btnLayout.addWidget(self.btn_preprocess)
        self.btnLayout.addWidget(self.btn_scan)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addLayout(self.btnLayout)
        self.window.setLayout(self.layout)

        # Kết nối sự kiện
        self.btn_open.clicked.connect(self.load_image)
        self.btn_preprocess.clicked.connect(self.preprocess_image)
        self.btn_scan.clicked.connect(self.DBScan)

        self.image_data = None

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self.window, "Chọn tệp DICOM", "", "DICOM Files (*.dcm)")
        if file_path:
            try:
                dicom = pydicom.dcmread(file_path)
                if hasattr(dicom, 'pixel_array'):
                    img = dicom.pixel_array.astype(np.uint8)
                    img = cv2.equalizeHist(img)
                    self.image_data = img
                    self.display_image(img)
                else:
                    raise ValueError("Tệp DICOM không chứa hình ảnh hợp lệ!")
            except Exception as e:
                print(f"Lỗi khi tải ảnh: {e}")

    def preprocess_image(self):
        if self.image_data is not None:
            try:
                # Làm mờ để giảm nhiễu
                img_blur = cv2.GaussianBlur(self.image_data, (5, 5), 0)
                # Áp dụng ngưỡng nhị phân
                _, img_thresh = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY)
                self.image_data = img_thresh
                self.display_image(img_thresh)
                print("Tiền xử lý ảnh thành công!")
            except Exception as e:
                print(f"Lỗi trong quá trình tiền xử lý: {e}")
        else:
            print("Không có ảnh để tiền xử lý!")

    def DBScan(self):
        if self.image_data is not None:
            try:
                _, binary = cv2.threshold(self.image_data, 127, 255, cv2.THRESH_BINARY)
                coords = np.column_stack(np.where(binary > 0))
                clustering = DBSCAN(eps=5, min_samples=10).fit(coords)
                labels = clustering.labels_

                plt.figure(figsize=(8, 8))
                unique_labels = set(labels)
                for label in unique_labels:
                    color = 'black' if label == -1 else plt.cm.jet(float(label) / max(unique_labels))
                    mask = (labels == label)
                    plt.scatter(coords[mask, 1], coords[mask, 0], c=[color], s=1)
                plt.title("Kết quả DBSCAN trên ảnh")
                plt.gca().invert_yaxis()
                plt.show()
            except Exception as e:
                print(f"Lỗi trong DBSCAN: {e}")
        else:
            print("Không có ảnh để phân tích DBSCAN!")

    def display_image(self, img):
        try:
            height, width = img.shape
            bytes_per_line = width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_img)
            self.label.setPixmap(pixmap)
            self.label.setScaledContents(True)
        except Exception as e:
            print(f"Lỗi khi hiển thị ảnh: {e}")

    def show(self):
        self.window.show()

app = QApplication([])
my_app = App()
my_app.show()
sys.exit(app.exec())
