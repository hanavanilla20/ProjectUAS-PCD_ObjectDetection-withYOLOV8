#detection dengan video input 

import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt') 

classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

video_path = 'videoKota.mp4' 
cap = cv2.VideoCapture(video_path)

highest_conf = 0.0
highest_conf_label = ""

while True:
    ret, img = cap.read()
    if not ret:
        break
    height, width, _ = img.shape

    results = model(img)

    
    for result in results:
        boxes = result.boxes.xyxy  
        confidences = result.boxes.conf  
        class_ids = result.boxes.cls 

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)  
            label = f"{classes[int(class_id)]} {conf:.2f}"
            color = (0, 255, 0) 
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            print(f"Detected {classes[int(class_id)]} with accuracy {conf:.2f}")

            if conf > highest_conf:
                highest_conf = conf
                highest_conf_label = classes[int(class_id)]

    cv2.imshow("Video Object Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nHighest accuracy: {highest_conf:.2f} for class: {highest_conf_label}")