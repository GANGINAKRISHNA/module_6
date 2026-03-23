import cv2
import numpy as np
import os

# Load your images
img1 = cv2.imread("images/img1.png")
img2 = cv2.imread("images/img2.png")
img3 = cv2.imread("images/img3.png")
img4 = cv2.imread("images/img4.png")

images = [img1, img2, img3, img4]

# Create output folder
os.makedirs("outputs/sfm", exist_ok=True)

# ORB feature detector
orb = cv2.ORB_create(3000)

def get_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des

def match_features(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches[:200]  # keep best matches

def compute_homography(img_ref, img):
    kp1, des1 = get_features(img_ref)
    kp2, des2 = get_features(img)

    matches = match_features(des1, des2)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

    return H, kp1, kp2, matches

# Use first image as reference
base_img = images[0]
h, w = base_img.shape[:2]

print("📌 Using img1.png as reference plane\n")

for i in range(1, 4):
    print(f"Processing img{i+1}.png...")

    H, kp1, kp2, matches = compute_homography(base_img, images[i])

    if H is None:
        print("❌ Homography failed")
        continue

    print("Homography Matrix:")
    print(H, "\n")

    # Warp image to align with reference
    warped = cv2.warpPerspective(images[i], H, (w, h))

    # Save warped image
    cv2.imwrite(f"outputs/sfm/warped_{i+1}.png", warped)

    # Draw matches (for report/demo)
    match_img = cv2.drawMatches(
        base_img, kp1,
        images[i], kp2,
        matches[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(f"outputs/sfm/matches_{i+1}.png", match_img)

print("✅ SfM processing completed!")