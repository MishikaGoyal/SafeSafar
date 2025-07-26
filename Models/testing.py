import cv2
from ultralytics import YOLO

# Load your custom trained model
model = YOLO("Models/yolov8n.pt")  # or provide full path

# Start webcam (0 is default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLOv8 prediction
    results = model.predict(source=frame, stream=True, conf=0.5)

    for r in results:
        # Draw boxes and labels
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box coords
            conf = float(box.conf[0])               # confidence
            cls_id = int(box.cls[0])
            print(cls_id)
            if(cls_id==1):
                break      #This if is for testing          # class id
            label = model.names[cls_id]             # class name

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label}: {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Print Drowsiness status
            # if "drowsy" in label.lower():
            #     print("⚠️ Person is Drowsy!")
            # else:
            #     print("✅ Person is Not Drowsy.")

    # Show the output frame
    cv2.imshow("YOLOv8 Live Drowsiness Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
