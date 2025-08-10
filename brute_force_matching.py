import cv2
import numpy as np

# Initialize ORB detector and BFMatcher
orb = cv2.ORB_create(
    nfeatures=5000,          # Keep more keypoints
    scaleFactor=1.1,         # Finer scale pyramid
    nlevels=50,              # More scale robustness
    edgeThreshold=100,        # Detect closer to image border
    WTA_K=4,                 # More robust BRIEF comparisons
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=20         # Detect more corners
)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # For ORB (binary descriptors)

# Load images (replace with your image paths)
img1 = cv2.imread('assets/notre_dame_2.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('assets/notre_dame_3.jpg', cv2.IMREAD_GRAYSCALE)

img1_resized = cv2.resize(img1, (int(img1.shape[1]*0.15),int(img1.shape[0]*0.15)), interpolation=cv2.INTER_AREA)
img2_resized = cv2.resize(img2, (int(img2.shape[1]*0.15),int(img2.shape[0]*0.15)), interpolation=cv2.INTER_AREA)

# Detect keypoints and compute descriptors
kp1, des1 = orb.detectAndCompute(img1_resized, None)
kp2, des2 = orb.detectAndCompute(img2_resized, None)

# Match descriptors
matches = bf.match(des1, des2)

# Sort them in the order of their distance
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 20 matches.
img3 = cv2.drawMatches(img1_resized,kp1,img2_resized,kp2,matches[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show resized image
cv2.imshow("Brute-Force Matching", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()