import cv2
import numpy as np
import mediapipe as mp
from matplotlib.testing.compare import crop_to_same
from mediapipe.python.solutions.selfie_segmentation import SelfieSegmentation
from mediapipe.tasks.python.components.containers import landmark

# MediaPipe Initializations
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_selfie = mp.solutions.selfie_segmentation

# Constants
HEAD_TOP_PADDING_FACTOR = 0.5
TORSO_PADDING_HEIGHT_FACTOR = 3.5
HORIZONTAL_PADDING_FACTOR = 1.0
NOSE_ALIGNMENT_Y_RATIO = 0.85

# Function: CreateColorBackground
# Purpose: Returns a background with a solid color
def createBackground(shape, colorChoice="grey"):
    """Creates a solid background image."""
    imageHeight, imageWidth, _ = shape

    if colorChoice == "white":
        color = (255, 255, 255)
    elif colorChoice == "grey":
        color = (180, 180, 180)
    elif colorChoice == "blue":
        color = (255, 0, 0)
    else:
        color = (0, 0, 0)

    return np.full((imageHeight, imageWidth, 3), color, dtype=np.uint8)

# Function: DetectAndBoxPeople()
# Purpose: Finds and create bounding boxes
def detectAndBoxPeople(inputImage):
    """Detects faces and generates padded bounding boxes for the torso and head"""
    imageHeight, imageWidth, _ = inputImage.shape
    personBoundingBoxes = []

    # Face Detecting Logic
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as faceDetector:
        results = faceDetector.process(cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB))

        if not results.detections:
            return personBoundingBoxes

        for detection in results.detections:
            relativeBoundingBox = detection.location_data.relative_bounding_box

            # Convert relative coordinates to pixel values
            xFace = int(relativeBoundingBox.xmin * imageWidth)
            yFace = int(relativeBoundingBox.ymin * imageHeight)
            wFace = int(relativeBoundingBox.width * imageWidth)
            hFace = int(relativeBoundingBox.height * imageHeight)

            # --- Calculate Padding based on factors ---
            paddingWidth = int(wFace * HORIZONTAL_PADDING_FACTOR)
            paddingHeightTorso = int(hFace * TORSO_PADDING_HEIGHT_FACTOR)
            paddingHeightHeadTop = int(hFace * HEAD_TOP_PADDING_FACTOR)

            # --- Apply Padding to create the final person bounding box ---
            xStart = max(0, xFace - (paddingWidth // 2))

            # Shift Y upwards to include the top of the head/hair
            yStart = max(0, yFace - paddingHeightHeadTop)

            # Final width and height calculation
            boxWidth = min(imageWidth - xStart, wFace + paddingWidth)
            boxHeight = min(imageHeight - yStart, hFace + paddingHeightTorso + paddingHeightHeadTop)

            personBoundingBoxes.append({'box': (xStart, yStart, boxWidth, boxHeight)})

    return personBoundingBoxes

# Function: AnalyzePersonHeight():
# Purpose: Finds the shortest person
def analyzePersonHeight(originalImage, personData):
    """
    Runs pose analysis on each person to determine their relative height
    and finds the minimum height amongst the group
    """
    minRelativeHeight = float('inf')
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as poseModel:
        for data in personData:
            x, y, w, h = data['box']
            regionOfInterest = originalImage[y:y + h, x:x + w]

            if regionOfInterest.size == 0: continue

            results = poseModel.process(cv2.cvtColor(regionOfInterest, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                def getRegionOfInterestPixelY(landmarkIndex):
                    # Denormalize to ROI pixel coordinates (0 to h)
                    return landmarks[landmarkIndex].y * regionOfInterest.shape[0]

                try:
                    # Top Reference: Nose (0)
                    noseY = getRegionOfInterestPixelY(mp_pose.PoseLandmark.NOSE)

                    # Bottom Reference: Mid-Hip
                    hipLeftY = getRegionOfInterestPixelY(mp_pose.PoseLandmark.LEFT_HIP)
                    hipRightY = getRegionOfInterestPixelY(mp_pose.PoseLandmark.RIGHT_HIP)
                    midHipY = (hipLeftY + hipRightY) / 2

                    # Relative Height calculation (Hip Y - Nose Y)
                    relativeHeight = midHipY - noseY

                    if (relativeHeight > 0):
                        data['relativeHeight'] = relativeHeight
                        data['noseYReference'] = noseY  # Used for alignment
                        minRelativeHeight = min(minRelativeHeight, relativeHeight)
                    else:
                        data['skip'] = True
                except Exception:
                    data['skip'] = True
            else:
                data['skip'] = True

        # Filter out skipped people and return the minimum height found
        personData = [d for d in personData if d.get('skip') is not True and d.get('relativeHeight') is not None]

        if minRelativeHeight == float('inf'):
            return [], 0  # Return empty list and zero height if no valid measurements

    return personData, minRelativeHeight

# Function: ScaleToShortestPerson()
# Readjusts the height of everyone so that they are the same height
def scaleToShortestPersonAndPaste(inputImagePath, backgroundImagePath):
    """
    Reads people from the input image, scales them to the shortest person's relative height,
    and then paste them into a new background.
    """

    originalImage = cv2.imread(inputImagePath, cv2.IMREAD_COLOR)
    finalCanvas = cv2.imread(backgroundImagePath, cv2.IMREAD_COLOR)

    if originalImage is None:
        print("Could not read image")
        return

    if finalCanvas is None:
        finalCanvas = createBackground(originalImage.shape)

    imageHeight, imageWidth, _ = originalImage.shape
    personData = detectAndBoxPeople(originalImage)

    # Find Shortest Person (Pose Analysis):
    personData, minRelativeHeight = analyzePersonHeight(originalImage, personData)

    if not personData:
        print("Could not detect person")
        return

    # Scale, Segment, and Paste
    alignmentYCanvas = int(finalCanvas.shape[0] * NOSE_ALIGNMENT_Y_RATIO)

    with mp_selfie.SelfieSegmentation(model_selection=1) as segmenter:
        for data in personData:
            x, y, w, h = data['box']
            originalRelativeHeight = data['relativeHeight']
            noseYReferenceRegionOfInterest = data['noseYReference']  # Nose Y coordinate relative to ROI top

            # Calculate Scaling Factor and New Dimensions
            scaleFactor = minRelativeHeight / originalRelativeHeight
            newWidth = int(w * scaleFactor)
            newHeight = int(h * scaleFactor)

            if newWidth <= 0 or newHeight <= 0: continue

            # Segment and Resize
            regionOfInterestOriginal = originalImage[y:y + h, x:x + w]
            segmentationResults = segmenter.process(cv2.cvtColor(regionOfInterestOriginal, cv2.COLOR_BGR2RGB))
            if segmentationResults.segmentation_mask is None: continue

            mask = (segmentationResults.segmentation_mask > 0.05).astype(np.uint8) * 255
            scaledPerson = cv2.resize(regionOfInterestOriginal, (newWidth, newHeight))
            scaledMask = cv2.resize(mask, (newWidth, newHeight))

            # Calculate Paste Position (Nose Alignment)

            scaledNoseDistanceToTop = int(noseYReferenceRegionOfInterest * scaleFactor)
            yDestination = alignmentYCanvas - scaledNoseDistanceToTop
            xDestination = x

            # Clamping and Compositing
            y1Destination, y2Destination = max(0, yDestination), min(imageHeight, yDestination + newHeight)
            x1Destination, x2Destination = max(0, xDestination), min(imageWidth, xDestination + newWidth)

            destinationPatch = finalCanvas[y1Destination:y2Destination, x1Destination:x2Destination]
            hDestination, wDestination = destinationPatch.shape[:2]

            finalPersonPart = cv2.resize(scaledPerson, (wDestination, hDestination))
            finalMask = cv2.resize(scaledMask, (wDestination, hDestination))

            # Blending
            inverseMask = cv2.bitwise_not(finalMask)
            backgroundPart = cv2.bitwise_and(destinationPatch, destinationPatch, mask=inverseMask)
            personPart = cv2.bitwise_and(finalPersonPart, finalPersonPart, mask=finalMask)
            combined = cv2.add(backgroundPart, personPart)

            finalCanvas[y1Destination:y2Destination, x1Destination:x2Destination] = combined

    # Display
    padding = 10
    spacer = np.zeros((imageHeight, padding, 3), dtype=np.uint8)

    stitchedImage = np.concatenate((originalImage, spacer, finalCanvas), axis=1)

    RESIZE_FACTOR = 0.5

    newWidth = int(stitchedImage.shape[1] * RESIZE_FACTOR)
    newHeight = int(stitchedImage.shape[0] * RESIZE_FACTOR)

    displayImage = cv2.resize(stitchedImage, (newWidth, newHeight))

    cv2.namedWindow('Comparison (Original vs. Processed)', cv2.WINDOW_NORMAL)
    cv2.imshow("Comparison (Original vs. Processed)", displayImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function: CropToShortestPersonAndInpaint()
# Purpose: [Deprecated Version] Finds the shortest person, segment them out of the picture,
# infill the background, and paste the images back into background but cropped with respect to smallest height
def cropToShortestPersonAndInpaint(inputImagePath):
    """
    [Deprecated] Detects people, align their heads to the lowest head, and inpaints the background to account
    for cutout sections
    """
    originalImage = cv2.imread(inputImagePath, cv2.IMREAD_COLOR)

    if originalImage is None:
        print("Could not read image")
        return

    imageHeight, imageWidth, _ = originalImage.shape

    # Create Bounding Boxes
    personData = detectAndBoxPeople(originalImage)

    if not personData:
        print("Could not detect person")
        return

    # Find the lowest head
    allYCoordinates = [data['box'][1] for data in personData] # Note: Should be box[1] for Y-coordinate
    targetYAlignment = max(allYCoordinates)
    print(f"Target head-level (lowest head) set to y={targetYAlignment}")

    # Setting up segmenter
    mp_selfie = mp.solutions.selfie_segmentation

    with mp_selfie.SelfieSegmentation(model_selection=1) as segmenter:
        masterMask = np.zeros(originalImage.shape[:2], dtype=np.uint8)

        for data in personData:
            x, y, w, h = data['box']
            padding = 20

            y1Roi = max(0, y-padding)
            y2Roi = min(imageHeight, y+h+padding)
            x1Roi = max(0, x-padding)
            x2Roi = min(imageWidth, x+w+padding)

            regionOfInterestOriginal = originalImage[y1Roi:y2Roi, x1Roi:x2Roi]
            if regionOfInterestOriginal.size == 0 : continue

            rgbRoi = cv2.cvtColor(regionOfInterestOriginal, cv2.COLOR_BGR2RGB)
            results = segmenter.process(rgbRoi)

            if results.segmentation_mask is None: continue

            personMask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
            # --- FIX: Corrected dsize argument to (Width, Height) ---
            personMask = cv2.resize(personMask, (regionOfInterestOriginal.shape[1], regionOfInterestOriginal.shape[0]))

            #combining masks into master mask
            existingMaskRoi = masterMask[y1Roi:y2Roi, x1Roi:x2Roi]
            combinedMask = cv2.bitwise_or(existingMaskRoi, personMask)
            masterMask[y1Roi:y2Roi, x1Roi:x2Roi] = combinedMask

        # Create inpainted background
        kernel = np.ones((10,10), np.uint8)
        dilatedMask = cv2.dilate(masterMask, kernel, iterations=1)

        inpaintedBacgkround = cv2.inpaint(originalImage, dilatedMask, 5,cv2.INPAINT_TELEA)
        finalCanvas = inpaintedBacgkround.copy()

        # Align heads
        for data in personData:
            x, y, w, h = data['box']
            regionOfInterestOriginal = originalImage[y:y+h, x:x+w]
            segmentationResults = segmenter.process(cv2.cvtColor(regionOfInterestOriginal, cv2.COLOR_BGR2RGB))
            if segmentationResults.segmentation_mask is None: continue

            finalPersonPart = regionOfInterestOriginal.copy()
            finalMask = (segmentationResults.segmentation_mask > 0.05).astype(np.uint8) * 255
            finalMask = cv2.resize(finalMask,(w,h))

            yDestination = targetYAlignment
            xDestination = x

            y1Destination, y2Destination = max(0, yDestination), min(imageHeight, yDestination+h)
            x1Destination, x2Destination = max(0, xDestination), min(imageWidth, xDestination + w)

            destinationPatch = finalCanvas[y1Destination:y2Destination, x1Destination:x2Destination]
            # --- FIX: Use .shape[:2] to get Height and Width ---
            hDestination, wDestination = destinationPatch.shape[:2]

            # Resize the parts to match the clamped patch size
            finalPersonPart = cv2.resize(finalPersonPart, (wDestination, hDestination))
            finalMask = cv2.resize(finalMask, (wDestination, hDestination))

            # Blending
            inverseMask = cv2.bitwise_not(finalMask)
            backgroundPart = cv2.bitwise_and(destinationPatch, destinationPatch, mask=inverseMask)
            personPart = cv2.bitwise_and(finalPersonPart, finalPersonPart, mask=finalMask)
            combined = cv2.add(backgroundPart, personPart)

            finalCanvas[y1Destination:y2Destination, x1Destination:x2Destination] = combined

    # Display
    RESIZE_FACTOR = 1
    padding = 10
    spacer = np.zeros((imageHeight, padding, 3), dtype=np.uint8)

    stitchedImage = np.concatenate((originalImage, spacer, finalCanvas), axis=1)

    new_width = int(stitchedImage.shape[1] * RESIZE_FACTOR)
    new_height = int(stitchedImage.shape[0] * RESIZE_FACTOR)

    displayImage = cv2.resize(stitchedImage, (new_width, new_height))

    cv2.namedWindow('Comparison (Original vs. Aligned Only)', cv2.WINDOW_NORMAL)
    cv2.imshow("Comparison (Original vs. Aligned Only)", displayImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()