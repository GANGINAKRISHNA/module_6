import numpy as np
import cv2

def lucas_kanade_track(prev_gray, next_gray, prev_pts, window_size=15):
    # Very small LK step: solve per-point u,v
    Ix = cv2.Sobel(prev_gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(prev_gray, cv2.CV_64F, 0, 1, ksize=3)
    It = next_gray.astype(np.float64) - prev_gray.astype(np.float64)

    u = np.zeros(prev_pts.shape[0])
    v = np.zeros(prev_pts.shape[0])

    half = window_size // 2
    for i, (x, y) in enumerate(prev_pts):
        x0 = int(max(x-half, 0))
        x1 = int(min(x+half+1, prev_gray.shape[1]))
        y0 = int(max(y-half, 0))
        y1 = int(min(y+half+1, prev_gray.shape[0]))

        Ix_win = Ix[y0:y1, x0:x1].reshape(-1)
        Iy_win = Iy[y0:y1, x0:x1].reshape(-1)
        It_win = It[y0:y1, x0:x1].reshape(-1)

        A = np.stack([Ix_win, Iy_win], axis=1)
        b = -It_win

        # least squares
        if A.shape[0] >= 2:
            nu, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            u[i] = nu[0]
            v[i] = nu[1]
        else:
            u[i] = 0
            v[i] = 0

    return u, v

# Example usage would involve extracting prev_pts and iterating frames
