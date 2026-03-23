import cv2
import numpy as np
import os


def process_video(video_path, output_path):
    print(f"\nTrying to open: {video_path}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Try opening video (FFMPEG helps with .avi files)
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print(f"❌ Error: Cannot open video file {video_path}")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("❌ Error: Could not read first frame.")
        return

    h, w = first_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 20.0

    print(f"✅ Opened successfully | FPS: {fps} | Resolution: {w}x{h}")

    # Process only 30 seconds
    max_frames = int(fps * 30)
    frame_count = 0

    # Video writer (MP4 output)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    print("🎬 Processing started... Press 'q' to stop early.")

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ End of video reached early.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Convert flow to HSV for visualization
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Overlay on original frame
        combined = cv2.addWeighted(frame, 0.7, flow_rgb, 0.3, 0)

        # Draw motion vectors (grid sampling)
        step = 16
        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = flow[y, x]
                cv2.arrowedLine(
                    combined,
                    (x, y),
                    (int(x + fx), int(y + fy)),
                    (0, 255, 0),
                    1,
                    tipLength=0.3
                )

        # Save sample frames for report
        if frame_count in [10, 50, 100]:
            sample_path = f"outputs/sample_{os.path.basename(video_path)}_{frame_count}.png"
            cv2.imwrite(sample_path, combined)
            print(f"📸 Saved sample frame: {sample_path}")

        # Write to output video
        out.write(combined)

        # Display window
        cv2.imshow("Optical Flow", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⏹️ Stopped manually.")
            break

        prev_gray = gray.copy()
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"✅ Finished! Output saved to: {output_path}")


if __name__ == "__main__":
    # 🔹 UPDATE THESE PATHS BASED ON YOUR FILES

    # Example 1 (MP4 files)
    process_video("data/video1.mp4", "outputs/flow_video1.mp4")
    process_video("data/video2.mp4", "outputs/flow_video2.mp4")

    # Example 2 (AVI files) → uncomment if needed
    # process_video("data/video1.avi", "outputs/flow_video1.mp4")
    # process_video("data/video2.avi", "outputs/flow_video2.mp4")

    # Example 3 (your uploaded file test)
    # process_video("/mnt/data/flow_video1.avi", "outputs/test_video2.mp4")