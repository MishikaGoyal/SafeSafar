import cv2
from ultralytics import YOLO
import pygame

# Initialize pygame mixer for playing sound
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("Models\\siren-alert-96052.mp3")  # replace with your own sound file

# Load the YOLOv8 model
model = YOLO("Models\\testing.py")  # Provide full path if needed

# Open the default webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("🚀 Webcam started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLOv8 prediction on the frame
    results = model.predict(source=frame, stream=True, conf=0.5)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label}: {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Print detection info
            print(f"🎯 Detected: {label} (ID: {cls_id}) | Confidence: {conf:.2f}")

            # Example: Trigger alarm for 'drowsy' or specific class
            if label=='1':
                print("⚠️ Alert Condition Met - Playing Sound!")
                pygame.mixer.Sound.play(alert_sound)

    # Show the result frame
    cv2.imshow("🔍 YOLOv11 Live Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
