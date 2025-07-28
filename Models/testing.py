import cv2
from ultralytics import YOLO

model = YOLO("yawn.pt") 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model.predict(source=frame, stream=True, conf=0.5)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            conf = float(box.conf[0])              
            cls_id = int(box.cls[0])
            label = model.names[cls_id]             

            print(f"Detected: {label} (Class {cls_id}) - Confidence: {conf:.3f} ({conf*100:.1f}%)")
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label}: {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if label == '1':
                print("⚠️ Person is Drowsy!")
            else:
                print("✅ Person is Not Drowsy.")

    cv2.imshow("YOLOv8 Live Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
