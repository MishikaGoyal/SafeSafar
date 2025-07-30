import cv2
from ultralytics import YOLO
import pygame
import time

# Initialize pygame mixer
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("C:\\Users\\PRATHAM\\OneDrive\\Desktop\\SafeSafar\\Models\\siren-alert-96052.mp3")
alert_channel = None  # Channel to manage playback

# Load the YOLO model
model = YOLO("C:\\Users\\PRATHAM\\OneDrive\\Desktop\\SafeSafar\\Models\\yawn.pt")

# Open webcam
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

    label_detected = False  # Reset for each frame

    results = model.predict(source=frame, stream=True, conf=0.5)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Draw box & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label}: {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            print(f"🎯 Detected: {label} (ID: {cls_id}) | Confidence: {conf:.2f}")

            # If label '1' is detected, set flag
            if label == '1':
                label_detected = True

    # Control alert sound
    if label_detected:
        if alert_channel is None or not alert_channel.get_busy():
            print("⚠️ Alert ON - Playing Sound")
            alert_channel = alert_sound.play(-1)  # Loop sound
    else:
        if alert_channel is not None and alert_channel.get_busy():
            print("✅ Alert OFF - Stopping Sound")
            alert_channel.stop()
            alert_channel = None

    # Show frame
    cv2.imshow("🔍 YOLOv11 Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

