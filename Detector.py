import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (nano version for speed)
model = YOLO('yolov8n.pt')

# Start capturing video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam could not be opened.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference with stream=True for better performance on video
    results = model(frame, stream=True)

    # Draw bounding boxes manually for each result
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])      # Box coordinates
            conf = float(box.conf[0])                   # Confidence
            cls = int(box.cls[0])                       # Class ID
            label = f"{model.names[cls]} {conf:.2f}"    # Class label

            if conf > 0.5:  # Only show if confidence > 50%
                # Bold red rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

                # Bold red label
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    # Show the output
    cv2.imshow("YOLOv8 Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
