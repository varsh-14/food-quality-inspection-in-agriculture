
import cv2
import numpy as np

# Open video or webcam
cap = cv2.VideoCapture('wildlife.mp4')  # or use 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and blur to reduce noise
    frame = cv2.resize(frame, (640, 480))
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    # Convert to HSV for color-based filtering
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Define color range to detect brownish/earth-tone animals
    lower_brown = np.array([5, 50, 50])
    upper_brown = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Clean up the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area > 1500:  # Filter out small objects
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)

            if 0.5 < aspect_ratio < 3.0:  # Rough animal shape filter
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 100), 2)
                cv2.putText(frame, "Possible Animal", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 2)

    # Show results
    cv2.imshow("Color + Contour Animal Detection", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()