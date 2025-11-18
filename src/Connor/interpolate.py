import cv2
import numpy as np
import sys


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread("test.jpeg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(20, 20))

# If no faces detected, exit with a message
if len(faces) == 0:
    print("No faces detected in image")
    sys.exit(0)

# Select the bounding box whose bottom (y + h) is the lowest in the image
# i.e., the face with the maximum y+h value
x, y, w, h = max(faces, key=lambda r: r[1] + r[3])

cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2) # Blue rectangle, thickness 2

# Display the image with bounding boxes
#cv2.imshow('Face Detection', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#head_roi = img[y : y + h, x : x + w].copy()



def stretch_roi_and_overwrite(image_path, x, y, w, h, new_height):
    """
    Stretches a specified ROI vertically using standard cv2.resize, 
    but keeps the original canvas size by overwriting content below the ROI.
    
    Args:
        image_path (str): Path to the input image file.
        x, y, w, h (int): Bounding box coordinates (top-left x, y, width, height).
        new_height (int): The target height for the ROI. Must be greater than 'h' 
                          to show a stretch effect.
        
    Returns:
        np.array: The final image on the original canvas size.
    """
    # 1. Load the image and check
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
        
    # Create a working copy of the original image (the canvas remains the same size)
    final_img = img.copy()
    
    # Check if the desired stretch exceeds the image boundary
    if y + new_height > final_img.shape[0]:
        print(f"Warning: New height ({y + new_height}) exceeds image canvas height ({final_img.shape[0]}). The stretch will be clipped.")

    # 2. Extract the Region of Interest (ROI)
    # Note: We must ensure we only extract the part that fits in the original canvas
    # The actual region extracted might be smaller than 'h' if it's at the bottom edge.
    h_actual = min(h, final_img.shape[0] - y)
    
    head_roi = img[y : y + h_actual, x : x + w].copy()
    
    # 3. Resize the ROI using cv2.resize
    # We keep the width (w) the same but stretch the height.
    stretched_roi = cv2.resize(
        head_roi, 
        (w, new_height), 
        interpolation=cv2.INTER_LINEAR
    )
    
    # 4. Paste the stretched ROI back into the original canvas.
    # Align the BOTTOM of the stretched ROI with the bottom of the original ROI in the image.
    # Compute destination end (bottom of original ROI within canvas)
    dst_end = y + h_actual
    # Desired start row for the stretched ROI so its bottom aligns with dst_end
    dst_start = dst_end - new_height

    # If the start is above the image, we need to skip the top rows of the stretched ROI
    src_start = 0
    if dst_start < 0:
        src_start = -dst_start
        dst_start = 0

    # How many rows we can actually paste on the canvas
    fit_h = min(new_height - src_start, final_img.shape[0] - dst_start)
    if fit_h > 0:
        final_img[dst_start : dst_start + fit_h, x : x + w] = stretched_roi[src_start:src_start + fit_h, :]

    return final_img


im = stretch_roi_and_overwrite("test.jpeg",x,y,w,h,500)

cv2.imshow("original", img)
cv2.imshow("test", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
