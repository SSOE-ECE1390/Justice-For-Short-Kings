# Bottom-Crop and Align Method method
# - Equalizes everyone's height by cropping everyone taller than then shortest people
# - The legs / lower portions of the tall people will be cropped out
# - Shift their position down so that everyone has the same height level in terms of head position.

# Best Approach:
# 1. Cut out all of the people from the original picture
# 2. Infill the sillohette so that a background remains
# 3. Modify the height of everyone relative to short king
# 4. Paste images back onto the background

import cv2
import numpy as np
import matplotlib.pyplot as plt
#import mediapipe as mp

# -- Load Image --
imageBGR = cv2.imread('Images/testImage.png', cv2.IMREAD_COLOR)
cleanBackground = imageBGR.copy()

# -- Run Bounding box finding logic --
dummyBoundingBox = [
    {'x': 250, 'y': 550, 'width': 200, 'height': 400}, # Red Box
    {'x': 470, 'y': 450, 'width': 210, 'height': 600}, # Blue Box
    {'x': 700, 'y': 500, 'width': 220, 'height': 500}  # Green Box
]

targetHeight = 400
targetYCoord = 550

# Remove images from background and then infill
#mask = np.zeros(imageBGR.shape[:2], dtype=np.uint8)

# for box in dummyBoundingBox:
#     x, y, w, h = box['x'], box['y'], box['width'], box['height']
#     cv2.rectangle(mask, (x-5, y-5), (x+w+5, y+h+5), (255), cv2.FILLED)
#
# inpaintedBackgroundBGR = cv2.inpaint(cleanBackground, mask, 3, cv2.INPAINT_TELEA)
# finalImageBGR = inpaintedBackgroundBGR.copy()

imageWithHoles = imageBGR.copy()
mask = np.zeros(imageBGR.shape[:2], dtype=np.uint8)

for box in dummyBoundingBox:
    x, y, w, h = box['x'], box['y'], box['width'], box['height']
    cv2.rectangle(mask, (x,y), (x+w, y+h), (255,255,255), cv2.FILLED)
    cv2.rectangle(imageWithHoles, (x,y), (x+w, y+h), (0,0,0), cv2.FILLED)

inpaintedBackgroundBGR = cv2.inpaint(imageWithHoles, mask, 3,cv2.INPAINT_TELEA)
finalImageBGR = inpaintedBackgroundBGR.copy()

for box in dummyBoundingBox:
    currentX, currentY = box['x'], box['y']
    currentWidth, currentHeight = box['width'], box['height']

    if currentHeight <= targetHeight:
        srcRegion = imageBGR[
            currentY: currentY + currentHeight,
            currentX : currentX + currentWidth
        ]

        finalImageBGR[currentY: currentY + currentHeight, currentX: currentX + currentWidth] = srcRegion

    else:
        srcYStart = currentY
        srcYEnd = currentY + targetHeight
        srcXStart = currentX
        srcXEnd = currentX + currentWidth

        srcRegion = imageBGR[srcYStart:srcYEnd, srcXStart:srcXEnd]

        destYStart = targetYCoord
        destYEnd = targetYCoord + targetHeight
        destXStart = currentX
        destXEnd = currentX + currentWidth

        finalImageBGR[destYStart:destYEnd, destXStart:destXEnd] = srcRegion

imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
inpaintedBackgroundRGB = cv2.cvtColor(inpaintedBackgroundBGR, cv2.COLOR_BGR2RGB)
finalImageRGB = cv2.cvtColor(finalImageBGR, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(15, 10))
plt.subplot(1,3,1); plt.imshow(imageRGB); plt.title('Original Image')
plt.subplot(1,3,2); plt.imshow(inpaintedBackgroundRGB); plt.title('Inpainted Image')
plt.subplot(1,3,3); plt.imshow(finalImageRGB); plt.title('Final Image')

plt.tight_layout()
plt.show()
