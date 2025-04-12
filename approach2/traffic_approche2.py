from ultralytics import YOLO
import cv2

# Load model
model = YOLO("runs/detect/road_signs_v1/weights/best.pt")

# Configuration
CONFIDENCE_THRESHOLD = 0.6
NMS_IOU_THRESHOLD = 0.45

# Centered ROI (smaller box)
frame_width, frame_height = 640, 480  # Default camera resolution (updated later)
ROI_WIDTH, ROI_HEIGHT = 400, 300      # Smaller ROI size
roi_x1 = (frame_width - ROI_WIDTH) // 2
roi_y1 = (frame_height - ROI_HEIGHT) // 2
roi_x2 = roi_x1 + ROI_WIDTH
roi_y2 = roi_y1 + ROI_HEIGHT

cap = cv2.VideoCapture(0)

# Get actual camera resolution
ret, test_frame = cap.read()
if ret:
    frame_height, frame_width = test_frame.shape[:2]
    # Recalculate centered ROI
    roi_x1 = (frame_width - ROI_WIDTH) // 2
    roi_y1 = (frame_height - ROI_HEIGHT) // 2
    roi_x2 = roi_x1 + ROI_WIDTH
    roi_y2 = roi_y1 + ROI_HEIGHT

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Could not access camera.")
        break

    # Apply centered ROI
    roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    # Run inference
    results = model(roi_frame, 
                   conf=CONFIDENCE_THRESHOLD,
                   iou=NMS_IOU_THRESHOLD,
                   verbose=False)

    # Draw semi-transparent centered ROI box
    overlay = frame.copy()
    cv2.rectangle(overlay, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
    alpha = 0.3  # Transparency factor
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Process detections
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            x1 += roi_x1  # Convert ROI coords to full frame
            y1 += roi_y1
            x2 += roi_x1
            y2 += roi_y1

            # Draw bounding box (green)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label at bottom
            label = f"{model.names[int(cls)]} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = y2 + h + 10 if y2 + h + 10 < frame_height else y1 - 10
            cv2.rectangle(frame, (x1, label_y - h - 10), (x1 + w, label_y), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Display
    cv2.imshow(f"YOLOv8 (ROI: {ROI_WIDTH}x{ROI_HEIGHT})", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()