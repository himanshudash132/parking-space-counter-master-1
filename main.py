import cv2        # OpenCV library for computer vision tasks
import matplotlib.pyplot as plt  # Matplotlib for plotting (not used here, but available)
import numpy as np  # NumPy for numerical operations

# Utility functions to get parking spot bounding boxes and check occupancy
from util import get_parking_spots_bboxes, empty_or_not


def calc_diff(im1, im2):
    """
    Calculate the absolute difference between the mean pixel values of two image crops.
    Used to detect significant changes in a parking spot between two frames.
    """
    # Compute average pixel intensity for each crop and take absolute difference
    return np.abs(np.mean(im1) - np.mean(im2))


# Paths to mask image and video file
mask = './mask_1920_1080.png'
video_path = './data/parking_1920_1080_loop.mp4'

# Read the mask image in grayscale (0 flag)
mask = cv2.imread(mask, 0)

# Open the video capture for processing frames
cap = cv2.VideoCapture(video_path)

# Identify connected components in the mask to locate parking spots
# connectedComponentsWithStats returns: num_labels, labels, stats, centroids
connected_components = cv2.connectedComponentsWithStats(mask, connectivity=4, ltype=cv2.CV_32S)

# Generate bounding boxes for each parking spot from connected component stats
spots = get_parking_spots_bboxes(connected_components)

# Initialize status and difference buffers for each spot
spots_status = [None for _ in spots]  # True=empty, False=occupied
diffs = [None for _ in spots]         # Difference metric between frames

# Variables to hold previous reference frame and frame counter
previous_frame = None
frame_nmr = 0
ret = True
step = 30  # Process every 30th frame to reduce computation

# Main loop: read frames until video ends
while ret:
    ret, frame = cap.read()  # Grab next frame

    # Every 'step' frames, if we have a previous frame, compute change diffs
    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot  # Unpack bounding box coordinates

            # Crop the current frame to the parking spot region
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            # Compute difference between current and previous crop
            diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

        # Print sorted diffs in descending order for debug
        print([diffs[j] for j in np.argsort(diffs)][::-1])

    # Every 'step' frames, decide which spots to re-evaluate occupancy
    if frame_nmr % step == 0:
        if previous_frame is None:
            # On the first check, evaluate all spots
            arr_ = range(len(spots))
        else:
            # On subsequent checks, only evaluate spots with significant change
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]

        # Update status for selected spots
        for spot_indx in arr_:
            x1, y1, w, h = spots[spot_indx]
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            # Determine if the spot is empty
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_indx] = spot_status

    # Update reference frame every 'step' frames
    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    # Draw bounding boxes and status on frame for display
    for spot_indx, spot in enumerate(spots):
        spot_status = spots_status[spot_indx]
        x1, y1, w, h = spot

        # Green rectangle if spot is empty, red if occupied
        color = (0, 255, 0) if spot_status else (0, 0, 255)
        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

    # Overlay a filled rectangle as background for text
    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    # Display count of available spots
    cv2.putText(
        frame,
        f'Available spots: {sum(spots_status)} / {len(spots_status)}',
        (100, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Show the annotated frame in a resizable window
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1  # Increment frame counter

# Release video capture and close display windows
cap.release()
cv2.destroyAllWindows()
