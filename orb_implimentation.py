import cv2

image = cv2.imread("assets/building_blocks.jpg", cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(
    nfeatures=5000,          # Keep more keypoints
    scaleFactor=1.1,         # Finer scale pyramid
    nlevels=40,              # More scale robustness
    edgeThreshold=100,        # Detect closer to image border
    WTA_K=4,                 # More robust BRIEF comparisons
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=10         # Detect more corners
)

keypoints, descriptors = orb.detectAndCompute(image, None)

image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0,255,0),flags=0)

cv2.imshow("ORB Features", image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()