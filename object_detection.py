import cv2
from ultralytics import YOLO

def detect_objects_with_boxes(video_source=0, model_path="yolov8n.pt"): 
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_source)
    assert cap.isOpened(), f"Cannot open video source: {video_source}"

    while True: 
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        for box in results.boxes: 
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 225, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.0, (0, 225, 0), 2)

        cv2.imshow("Detected Objects", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": 
    detect_objects_with_boxes()
