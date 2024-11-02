import numpy as np
import cv2
import matplotlib.pyplot as plt

# Helper function to display images
def display_image(img, title="Image", cmap=None):
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Build PCA model for yellow color from target image
def build_color_model(image_path):
    image = cv2.imread(image_path).astype(np.float64) / 255.0
    color_obs = np.vstack([image[:, :, i].reshape(1, -1) for i in range(3)])
    
    # Mean and Covariance for the color model
    mean = np.mean(color_obs, axis=1).reshape(-1, 1)
    centered_data = color_obs - mean
    cov_matrix = np.cov(centered_data)

    return mean, np.linalg.inv(cov_matrix)  # Return inverse covariance for Mahalanobis distance

# Load target yellow image and build color model
mean_yellow, inv_cov_yellow = build_color_model('/home/kamal/CVPR_Labs/Lab_2/target_yellow.bmp')

# Load the test image
test_image = cv2.imread('/home/kamal/CVPR_Labs/Lab_2/kitchen.bmp').astype(np.float64) / 255.0
test_obs = np.vstack([test_image[:, :, i].reshape(1, -1) for i in range(3)])

# Compute Mahalanobis distance for each pixel in the test image
xsub = test_obs - mean_yellow
mahalanobis_dist_squared = np.sum((xsub.T @ inv_cov_yellow) * xsub.T, axis=1)
mahalanobis_dist = np.sqrt(mahalanobis_dist_squared)

# Reshape distance to match the image dimensions
distance_map = mahalanobis_dist.reshape(test_image.shape[:2])

# Normalize and display the distance map
normalized_map = distance_map / np.max(distance_map)
display_image(normalized_map, "Mahalanobis Distance Map (Yellow)", cmap=None)

# Apply threshold to highlight pixels similar to yellow
threshold_map = (distance_map < 3)  # 3 standard deviations as threshold
display_image(threshold_map, "Yellow Segmentation", cmap='gray')
