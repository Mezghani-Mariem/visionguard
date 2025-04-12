import cv2
import os
import numpy as np

# Create a folder to store your dataset
dataset_folder = "large_equilateral_triangle_road_signs"
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# Start webcam
cap = cv2.VideoCapture(0)

# Create a counter for the images
img_counter = 0

# Set a minimum area threshold for the triangle to be considered large enough
min_area = 500  # Adjust this value based on your requirements

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If the polygon has 3 vertices, it's a triangle
        if len(approx) == 3:
            # Calculate the lengths of the sides of the triangle
            side1 = np.linalg.norm(approx[0] - approx[1])
            side2 = np.linalg.norm(approx[1] - approx[2])
            side3 = np.linalg.norm(approx[2] - approx[0])

            # Check if the sides are approximately equal (tolerance can be adjusted)
            if abs(side1 - side2) < 5 and abs(side2 - side3) < 5:
                # Calculate the area of the triangle
                area = cv2.contourArea(contour)

                # Check if the area is large enough
                if area > min_area:
                    # Get the bounding box of the triangle
                    x, y, w, h = cv2.boundingRect(approx)
                    
                    # Crop and save the detected triangle region
                    img_counter += 1
                    triangle = frame[y:y + h, x:x + w]
                    cv2.imwrite(f"{dataset_folder}/equilateral_triangle_{img_counter}.jpg", triangle)

    # Show the frame with triangle detections
    cv2.imshow('Large Equilateral Triangle Road Sign Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
