import cv2
from ultralytics import YOLO

# Load YOLOv8
model = YOLO('yolov8n.pt')  # Menggunakan model YOLOv8 kecil, bisa diganti dengan model yang lebih besar

# Load coco.names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define video capture
cap = cv2.VideoCapture(0)  # 0 adalah id untuk kamera default

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLOv8 inference
    results = model(frame)

    # Showing information on the screen
    for result in results:
        boxes = result.boxes.xyxy  # Bounding boxes in (x1, y1, x2, y2) format
        confidences = result.boxes.conf  # Confidence scores
        class_ids = result.boxes.cls  # Class IDs

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
            label = f"{classes[int(class_id)]} {conf:.2f}"
            color = (0, 255, 0)  # You can set different colors for different classes
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
