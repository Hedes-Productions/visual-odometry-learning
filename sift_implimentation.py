import cv2

# Load the image
image = cv2.imread("assets/building_blocks.jpg", cv2.IMREAD_GRAYSCALE)

# Create a SIFT detector object
sift = cv2.SIFT_create(
    nfeatures=0,             # 0 = no limit on number of keypoints
    nOctaveLayers=12,         # Number of layers per octave
    contrastThreshold=0.08,  # Lower → more keypoints, including weak features
    edgeThreshold=10,        # Lower → detect closer to edges
    sigma=1.1                 # Gaussian blur sigma
)

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(image, None)

# Draw the keypoints on the image (optional)
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)

# Display the image with keypoints
cv2.imshow("SIFT Features", image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()