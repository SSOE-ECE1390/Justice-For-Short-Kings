import cv2
import numpy as np
import mediapipe as mp
from matplotlib.testing.compare import crop_to_same
from mediapipe.tasks.python.components.containers import landmark

# Initialize Solutions:
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_selfie = mp.solutions.selfie_segmentation

def create_background (shape, color_choice = "white"):
    h, w, _ = shape

    if color_choice == "white":
        color = (255, 255, 255)
    elif color_choice == "grey":
        color = (180, 180, 180)
    elif color_choice == "blue":
        color = (255, 0, 0)
    else:
        color = (0,0,0)

    return np.full((h,w,3), color, dtype=np.uint8)

def Scale_to_Shortest_Person(image_path, backgroundImagePath):

    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    imgHeight, imgWidth, _ = original_image.shape
    personData = []
    min_height = float('inf')

    FACE_PADDING_H_FACTOR = 3.5
    FACE_PADDING_W_FACTOR = 1.0
    ALIGNMENT_Y_RATIO = 0.85

    # Detect and Create Bounding Box between each human
    with mp_face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.7) as face_detection:
        results = face_detection.process(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

        if not results.detections:
            print("No Faces Found")
            return

        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            xFace = int(bboxC.xmin * imgWidth)
            yFace = int(bboxC.ymin * imgHeight)
            wFace = int(bboxC.width * imgWidth)
            hFace = int(bboxC.height * imgHeight)

            HEAD_TOP_PADDING_FACTOR = 0.5
            paddingWidth = int(wFace * FACE_PADDING_W_FACTOR)
            paddingHeight = int(hFace * FACE_PADDING_H_FACTOR)

            top_padding = int(hFace*HEAD_TOP_PADDING_FACTOR)

            x = max(0, xFace - (paddingWidth//2))
            y = max(0, yFace - top_padding)
            w = min(imgWidth - x, wFace + paddingWidth)
            h = min(imgHeight - y, hFace + paddingHeight + top_padding)

            personData.append({'box': (x,y,w,h)})

    # Find the height for each person and the shortest height
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose_model:
        for data in personData:
            x,y,w,h = data['box']
            roi = original_image[y:y+h, x:x+w]

            if roi.size == 0: continue

            results = pose_model.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                def get_roi_pixel_y(lmk_index):
                    return landmarks[lmk_index].y * roi.shape[0]

                try:
                    noseY = get_roi_pixel_y(mp_pose.PoseLandmark.NOSE)

                    hipLeftY = get_roi_pixel_y(mp_pose.PoseLandmark.LEFT_HIP)
                    hipRightY = get_roi_pixel_y(mp_pose.PoseLandmark.RIGHT_HIP)
                    midHipY = (hipLeftY + hipRightY) / 2

                    relativeHeight = midHipY - noseY

                    if (relativeHeight > 0):
                        data['height'] = relativeHeight
                        data['top_y_ref'] = noseY
                        data['bottom_y_ref'] = midHipY
                        min_height = min(min_height, relativeHeight)
                    else:
                        print(f"Negative Height detected for person at ({x},{y}). Skipped")
                        data['skip'] = True

                except Exception as e:
                    print(f"Error processing pose for person at ({x},{y}): {e}")
                    data['skip'] = True
            else:
                data['skip'] = True

        personData = [d for d in personData if d.get('skip') is not True and d.get('height') is not None]

        if min_height == float('inf'):
            print("No valid height measured.")
            return

    # Scale and resize all of the tall people
    #final_canvas = create_background(original_image.shape, background_color)
    final_canvas = cv2.imread(backgroundImagePath, cv2.IMREAD_COLOR)
    ALIGNMENT_Y_CANVAS = int(final_canvas.shape[0] * ALIGNMENT_Y_RATIO)

    with mp_selfie.SelfieSegmentation(model_selection=1) as segmenter:

        for data in personData:
            x,y,w,h = data['box']
            original_height = data['height']
            bottom_y_ref_roi = data['bottom_y_ref']

            scale_factor = min_height/original_height
            new_width = int(w*scale_factor)
            new_height = int(h*scale_factor)

            if new_width<=0 or new_height<=0: continue

            roi_original = original_image[y:y+h, x:x+w]
            rgbRoi = cv2.cvtColor(roi_original, cv2.COLOR_BGR2RGB)
            results = segmenter.process(rgbRoi)
            if results.segmentation_mask is None: continue

            mask = (results.segmentation_mask > 0.05).astype(np.uint8) * 255

            scaled_person = cv2.resize(roi_original, (new_width, new_height))
            scaled_mask = cv2.resize(mask, (new_width, new_height))

            # hip_distance_from_top_roi = bottom_y_ref_roi
            # scaled_hip_distance_from_top = int(hip_distance_from_top_roi*scale_factor)

            nose_distance_from_top_roi = data['top_y_ref']
            scaled_nose_distance_from_top = int(nose_distance_from_top_roi * scale_factor)

            y_dest = ALIGNMENT_Y_CANVAS - scaled_nose_distance_from_top
            x_dest = x

            y1_dest, y2_dest = max(0,y_dest), min(imgHeight, y_dest + new_height)
            x1_dest, x2_dest = max(0,x_dest), min(imgWidth, x_dest + new_width)

            dest_patch = final_canvas[y1_dest:y2_dest, x1_dest:x2_dest]
            h_dest, w_dest = dest_patch.shape[:2]

            final_person_part = cv2.resize(scaled_person, (w_dest, h_dest))
            final_mask = cv2.resize(scaled_mask, (w_dest, h_dest))

            inverseMask = cv2.bitwise_not(final_mask)
            bg_part = cv2.bitwise_and(dest_patch, dest_patch, mask=inverseMask)
            personPart = cv2.bitwise_and(final_person_part, final_person_part, mask = final_mask)

            combined = cv2.add(bg_part, personPart)
            final_canvas[y1_dest:y2_dest, x1_dest:x2_dest] = combined

    cv2.imshow('Original Image', original_image)
    cv2.imshow("Height Equalized Group Photo", final_canvas)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    IMAGE_FILE = "Images/imageProcessing.jpg"

    BACKGROUND_IMAGEPATH = "Background_Images/background_photo1.png"

    Scale_to_Shortest_Person(IMAGE_FILE, BACKGROUND_IMAGEPATH)