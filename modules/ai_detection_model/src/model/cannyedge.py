import cv2
import numpy as np

def apply_canny_edge_detection(image, threshold1, threshold2):
    # Convert to grayscale
    image = cv2.imread(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Apply Canny edge detector
    edges = cv2.Canny(blurred_image, threshold1, threshold2)
    return edges 


valid_stamp = '../../../datasets_stamp/train/stamp_001.png'
invalid_stamp = '../../../datasets_stamp/train/stamp_0008.png'
# Assuming `stamp_region` is the cropped region of the detected stamp
stamp_edges = apply_canny_edge_detection(invalid_stamp, 100, 200)


# Assuming `proper_stamp_image` is your reference proper stamp image
proper_stamp_edges = apply_canny_edge_detection(valid_stamp, 100, 200)


def compare_edge_density(edge_image1, edge_image2):
    # Calculate the density of edges (percentage of edge pixels)
    density1 = np.sum(edge_image1 > 0) / np.prod(edge_image1.shape)
    density2 = np.sum(edge_image2 > 0) / np.prod(edge_image2.shape)
    # Compare densities; you might want to define a threshold for "similar" densities
    return abs(density1 - density2)


# Compare the stamp with the proper stamp
difference = compare_edge_density(stamp_edges, proper_stamp_edges)
print("Difference in edge density:", difference)

# Assuming `difference` holds the calculated difference in edge density
threshold = 0.004  # Example threshold, adjust based on your dataset analysis

if difference <= threshold:
    print("The stamp is considered valid.")
else:
    print("The stamp is considered invalid.") 

proper_stamp_edges_resized = cv2.resize(proper_stamp_edges, (stamp_edges.shape[1], stamp_edges.shape[0]))

# Apply template matching
result = cv2.matchTemplate(stamp_edges, proper_stamp_edges_resized, cv2.TM_CCOEFF_NORMED)


# Get the maximum match value
max_match_value = np.max(result)
print("Maximum match value:", max_match_value)

match_threshold = 0.7  # Example threshold, adjust based on your analysis

if max_match_value > match_threshold:
    print("The stamp is considered valid.")
else:
    print("The stamp is considered invalid.")




