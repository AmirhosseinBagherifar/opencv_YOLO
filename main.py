import sys
import os
import cv2
import numpy as np
import csv
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QVBoxLayout, QFileDialog, QMessageBox
)
class PeopleCounterYOLO(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO")
        self.setGeometry(100, 100, 500, 250)

        self.open_image_button = QPushButton("انتخاب عکس")
        self.open_image_button.clicked.connect(self.open_image)

        self.save_csv_button = QPushButton("ذخیره در CSV")
        self.save_csv_button.clicked.connect(self.save_csv)
        self.save_csv_button.setEnabled(False) 

        self.file_label = QLabel(" فایلی انتخاب نشده ")
        self.count_label = QLabel("تعداد افراد: 0")

        layout = QVBoxLayout()
        layout.addWidget(self.open_image_button)
        layout.addWidget(self.file_label)
        layout.addWidget(self.count_label)
        layout.addWidget(self.save_csv_button)
        self.setLayout(layout)

        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getUnconnectedOutLayersNames()
        self.last_file = None
        self.last_count = 0

    def detect_people(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.layer_names)

        boxes, confidences = [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = scores[class_id]
                if self.classes[class_id] == "person" and confidence > 0.5:
                    center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype("int")
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        final_boxes = [boxes[i] for i in indexes.flatten()] if len(indexes) > 0 else []
        return final_boxes

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "انتخاب عکس", "", "Images (*.png *.jpg *.jpeg)")
        if not file_path:
            return

        self.file_label.setText(f"📁 عکس انتخابی: {file_path}")
        self.count_label.setText("در حال پردازش عکس...")

        image = cv2.imread(file_path)
        boxes = self.detect_people(image)

        for (x, y, w, h) in boxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        count = len(boxes)
        self.count_label.setText(f"تعداد افراد در عکس: {count}")
        cv2.imshow("YOLO - عکس", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.last_file = file_path
        self.last_count = count
        self.save_csv_button.setEnabled(True)

    def save_csv(self):
        if not self.last_file:
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "ذخیره CSV", "people_count.csv", "CSV Files (*.csv)")
        if not save_path:
            return

        file_exists = os.path.exists(save_path)

        try:
            with open(save_path, mode="a", newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["نام فایل" ,  "تعداد افراد"])
                writer.writerow([os.path.basename(self.last_file), self.last_count])
            QMessageBox.information(self,f"اضافه شد:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, f"خطا در ذخیره فایل:\n{e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PeopleCounterYOLO()
    window.show()
    sys.exit(app.exec_())
