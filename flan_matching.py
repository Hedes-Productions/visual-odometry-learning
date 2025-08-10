import cv2
import numpy as np

# Initialize ORB detector
orb = cv2.ORB_create(
    nfeatures=2000,          # Keep more keypoints
    scaleFactor=1.1,         # Finer scale pyramid
    nlevels=50,              # More scale robustness
    edgeThreshold=100,        # Detect closer to image border
    WTA_K=4,                 # More robust BRIEF comparisons
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=20         # Detect more corners
)

# Load images (replace with your image paths)
img1 = cv2.imread('assets/notre_dame_2.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('assets/notre_dame_3.jpg', cv2.IMREAD_GRAYSCALE)

img1_resized = cv2.resize(img1, (0,0),fx=0.13,fy=0.13, interpolation=cv2.INTER_AREA)
img2_resized = cv2.resize(img2, (0,0),fx=0.13,fy=0.13, interpolation=cv2.INTER_AREA)

# Detect keypoints and compute descriptors
kp1, des1 = orb.detectAndCompute(img1_resized, None)
kp2, des2 = orb.detectAndCompute(img2_resized, None)

# FLANN parameters
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=50)   # or pass empty dictionary

# Initialize FLANN matcher
flann = cv2.FlannBasedMatcher(index_params,search_params)

# Match descriptors
matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# Ratio test as per Lowe's paper
for i, match in enumerate(matches):
    if len(match) == 2:
        m, n = match
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)

img3 = cv2.drawMatchesKnn(img1_resized,kp1,img2_resized,kp2,matches,None,**draw_params)

cv2.imshow("FLANN Matching", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()