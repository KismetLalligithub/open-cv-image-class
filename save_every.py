import cv2
import os
import time
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
VIDEO_SOURCE = 0
SAVE_EVERY = 15
OUT_DIR = Path("output")

def detect_save_every_n_seconds(): 
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    assert cap.isOpened(), f"Cannot open video source: {VIDEO_SOURCE}"
    OUT_DIR.mkdir(exist_ok=True)

    last_save_ts = 0

    print("Press 'q' to quit.")

    frame_idx = 0 
    while True: 
        ret, frame = cap.read()
        if not ret: 
            break

        results = model(frame)[0]

        for box in results.boxes: 
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 225, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 225, 0), 2)
        
        now = time.time()
        if now - last_save_ts >= SAVE_EVERY and len(results.boxes) > 0: 
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            for i, box in enumerate(results.boxes): 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                crop = frame[y1:y2, x1:x2]

                filename = OUT_DIR / f"{timestamp}_{frame_idx}_{label}_{i}.jpg"
                cv2.imwrite(str(filename), crop)
            print(f"[{timestamp}] Saved {len(results.boxes)} objects(s).")
            last_save_ts = now
            frame_idx += 1 
        
        cv2.imshow("YOLO Detection (saving every 15 s)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": 
    detect_save_every_n_seconds()
