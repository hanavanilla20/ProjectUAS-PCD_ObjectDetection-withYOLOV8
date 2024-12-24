import cv2
import numpy as np
from ultralytics import YOLO

# Memuat model YOLO
model = YOLO('yolov8n.pt') 

# Membaca nama kelas dari file coco.names
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# Memuat gambar
image_path = 'imageKota.jpg'  # Ganti dengan path gambar Anda
img = cv2.imread(image_path)

# Menjalankan deteksi objek pada gambar
results = model(img)

highest_conf = 0.0
highest_conf_label = ""

for result in results:
    boxes = result.boxes.xyxy  # Bounding box koordinat
    confidences = result.boxes.conf  # Kepercayaan
    class_ids = result.boxes.cls  # ID kelas

    for box, conf, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)  
        label = f"{classes[int(class_id)]} {conf:.2f}"
        color = (0, 255, 0)  # Warna kotak (hijau)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        print(f"Detected {classes[int(class_id)]} with accuracy {conf:.2f}")

        if conf > highest_conf:
            highest_conf = conf
            highest_conf_label = classes[int(class_id)]

# Menampilkan gambar dengan kotak bounding
cv2.imshow("Image Object Detection", img)
cv2.waitKey(0)  # Tunggu hingga tombol ditekan
cv2.destroyAllWindows()

print(f"\nHighest accuracy: {highest_conf:.2f} for class: {highest_conf_label}")
