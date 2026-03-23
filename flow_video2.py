import cv2
import numpy as np
import os

# Input and Output Paths
input_path = "Data/video2.mp4"
output_path = "outputs/flow_video2.avi"

# Create output folder if not exists
os.makedirs("outputs", exist_ok=True)

# Load video
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print("Error: Cannot open video2")
    exit()

# Read first frame
ret, frame1 = cap.read()
if not ret:
    print("Error: Cannot read first frame")
    exit()

# Convert to grayscale
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# HSV for visualization
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(
    output_path,
    fourcc,
    20.0,
    (frame1.shape[1], frame1.shape[0])
)

print("Processing video2...")

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Optical Flow
    flow = cv2.calcOpticalFlowFarneback(
        prvs, next_frame, None,
        0.5, 5, 25, 5, 7, 1.5, 0
    )

    # Convert to magnitude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Save frame
    out.write(bgr)

    # Optional display
    cv2.imshow("Optical Flow - Video2", bgr)
    if cv2.waitKey(25) & 0xFF == 27:
        break

    prvs = next_frame

cap.release()
out.release()
cv2.destroyAllWindows()

print("Output saved at:", output_path)