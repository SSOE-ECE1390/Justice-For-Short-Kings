import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread("test.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

x = face[0][0]
y = face[0][1]
w = face[0][2]
h = face[0][3]

#cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2) # Blue rectangle, thickness 2

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
    
    # 4. Paste the stretched ROI back into the original canvas
    # The new ROI starts at the original 'y' and extends to 'y + new_height'
    
    # Determine the section of the stretched ROI that actually fits on the canvas
    fit_h = min(new_height, final_img.shape[0] - y)

    # Paste the fitting part of the stretched ROI back
    final_img[y : y + fit_h, x : x + w] = stretched_roi[0:fit_h, :]

    return final_img


im = stretch_roi_and_overwrite("test.jpg",x,y,w,h,200)

cv2.imshow("test", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
