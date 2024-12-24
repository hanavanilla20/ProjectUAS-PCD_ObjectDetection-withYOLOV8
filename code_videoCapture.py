import cv2
import numpy as np
from ultralytics import YOLO

# Memuat model YOLO
model = YOLO('yolov8n.pt') 

# Membaca nama kelas dari file coco.names
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# Menggunakan kamera sebagai input (kamera default: index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Tidak dapat mengakses kamera.")
    exit()

highest_conf = 0.0
highest_conf_label = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Menjalankan deteksi objek pada frame
    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy  # Bounding box koordinat
        confidences = result.boxes.conf  # Kepercayaan
        class_ids = result.boxes.cls  # ID kelas

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)  
            label = f"{classes[int(class_id)]} {conf:.2f}"
            color = (0, 255, 0)  # Warna kotak (hijau)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            print(f"Detected {classes[int(class_id)]} with accuracy {conf:.2f}")

            if conf > highest_conf:
                highest_conf = conf
                highest_conf_label = classes[int(class_id)]

    # Menampilkan frame dengan deteksi objek
    cv2.imshow("Kamera Object Detection", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nHighest accuracy: {highest_conf:.2f} for class: {highest_conf_label}")
