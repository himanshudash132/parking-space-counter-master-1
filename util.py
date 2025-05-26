import pickle  # For serializing and deserializing Python objects

from skimage.transform import resize  # To resize image arrays for model input
import numpy as np  # Numerical operations on arrays
import cv2  # OpenCV library for image processing

# Constants representing parking spot states
EMPTY = True
NOT_EMPTY = False

# Load pre-trained classification model from disk
# 'model.p' should be a pickle file containing a trained classifier
MODEL = pickle.load(open("model.p", "rb"))


def empty_or_not(spot_bgr):
    """
    Determine if a given parking spot crop is empty or occupied.

    Parameters:
    - spot_bgr: NumPy array of shape (H, W, 3), the BGR image of the parking spot.

    Returns:
    - EMPTY (True) if the spot is classified as empty.
    - NOT_EMPTY (False) if the spot is classified as occupied.
    """
    # Prepare a list to hold flattened image data
    flat_data = []

    # Resize the spot image to a fixed 15x15 resolution with 3 color channels
    img_resized = resize(spot_bgr, (15, 15, 3))
    # Flatten the 3D image into a 1D feature vector
    flat_data.append(img_resized.flatten())

    # Convert list to NumPy array of shape (1, 15*15*3)
    flat_data = np.array(flat_data)

    # Use the loaded model to predict occupancy; output is 0 or 1
    y_output = MODEL.predict(flat_data)

    # Map model output to our EMPTY/NOT_EMPTY constants
    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY


def get_parking_spots_bboxes(connected_components):
    """
    Extract bounding boxes for each detected parking spot from connected component stats.

    Parameters:
    - connected_components: tuple returned by cv2.connectedComponentsWithStats,
      containing (num_labels, label_ids, stats, centroids).

    Returns:
    - List of [x, y, width, height] for each parking spot region.
    """
    # Unpack connected component output
    totalLabels, label_ids, stats, centroids = connected_components

    slots = []  # List to accumulate bounding boxes
    coef = 1    # Scaling coefficient (if mask was resized before)

    # Skip the first label (background), iterate over each component
    for i in range(1, totalLabels):
        # Extract left, top, width, and height from stats
        x1 = int(stats[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(stats[i, cv2.CC_STAT_TOP] * coef)
        w  = int(stats[i, cv2.CC_STAT_WIDTH] * coef)
        h  = int(stats[i, cv2.CC_STAT_HEIGHT] * coef)

        # Append bounding box for this spot
        slots.append([x1, y1, w, h])

    return slots
