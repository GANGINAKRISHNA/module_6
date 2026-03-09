import cv2
import numpy as np

def main(video_path, output_path="flow_output.avi"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    ret, prev = cap.read()
    if not ret:
        return
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    h, w = prev_gray.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        # Convert flow to visualization
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv = np.zeros_like(frame)
        hsv[...,1] = 255

        # Hue represents direction
        hsv[...,0] = ang * 180 / np.pi / 2
        # Value represents magnitude
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Overlay
        vis = cv2.addWeighted(frame, 0.8, flow_vis, 0.6, 0)
        out.write(vis)

        cv2.imshow('flow', vis)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        prev_gray = gray.copy()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("data/video1.mp4", "flow_video1.avi")
