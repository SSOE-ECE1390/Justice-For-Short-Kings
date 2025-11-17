import cv2
import numpy as np
import mediapipe as mp
from PIL.ImageOps import scale
from mediapipe.python.solutions.drawing_styles import get_default_face_mesh_iris_connections_style


def segment_crop_and_align(original_image, bounding_boxes, background_image, segmenter, y_target):
    """
    Equalizes head height by pasting all segmented people at the same y_target.
    No height-based cropping is performed; "taller" people will run off the bottom.
    """
    canvas = background_image.copy()
    print("\n--- Starting Final Compositing ---")

    for box in bounding_boxes:
        (x, y, w, h) = box

        # --- 1. Get the FULL ROI and FULL MASK ---
        padding = 20
        y1_roi = max(0, y - padding)
        y2_roi = min(original_image.shape[0], y + h + padding)
        x1_roi = max(0, x - padding)
        x2_roi = min(original_image.shape[1], x + w + padding)

        roi_original = original_image[y1_roi:y2_roi, x1_roi:x2_roi]

        if roi_original.size == 0:
            print(f"Skipping empty ROI at (x,y) = {x},{y}")
            continue

        rgb_roi = cv2.cvtColor(roi_original, cv2.COLOR_BGR2RGB)
        results = segmenter.process(rgb_roi)

        if results.segmentation_mask is None:
            print(f"Warning: Segmenter found no person in box {box}")
            continue

        mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (roi_original.shape[1], roi_original.shape[0]))

        # --- We are now using the FULL person and FULL mask (no cropping) ---
        full_person_roi = roi_original
        full_mask = mask  # This is our 1-channel mask

        if full_person_roi.size == 0 or full_mask.size == 0:
            print(f"Skipping empty full ROI at (x,y) = {x},{y}")
            continue

        # --- 2. Define Destination and Paste ---
        try:
            h_person, w_person = full_person_roi.shape[:2]

            # --- 3. Define Destination Coordinates ---
            # We use the 'x' from the box, but the 'y' from our alignment target
            x1_dest = x
            y1_dest = y_target  # ALIGN ALL HEADS

            x2_dest = x1_dest + w_person
            y2_dest = y1_dest + h_person

            # --- 4. Clamp coordinates to be INSIDE the canvas ---
            if x1_dest < 0: x1_dest = 0
            if y1_dest < 0: y1_dest = 0
            if x2_dest > canvas.shape[1]: x2_dest = canvas.shape[1]
            if y2_dest > canvas.shape[0]: y2_dest = canvas.shape[0]

            # --- 5. Get the destination patch and its REAL shape ---
            dest_patch = canvas[y1_dest:y2_dest, x1_dest:x2_dest]

            if dest_patch.size == 0:
                print(f"  SKIPPING: Destination patch is empty.")
                continue

            h_dest, w_dest = dest_patch.shape[:2]

            # --- 6. Resize person/mask to fit the (clamped) destination ---
            if (h_person, w_person) != (h_dest, w_dest):
                full_person_roi = cv2.resize(full_person_roi, (w_dest, h_dest))
                full_mask = cv2.resize(full_mask, (w_dest, h_dest))  # Resize 1-channel mask

            # --- 7. Your compositing logic (now safe to run) ---
            inverseMask = cv2.bitwise_not(full_mask)  # 1-channel
            bg_part = cv2.bitwise_and(dest_patch, dest_patch, mask=inverseMask)  # 1-channel mask
            personPart = cv2.bitwise_and(full_person_roi, full_person_roi, mask=full_mask)  # 1-channel mask

            combined = cv2.add(bg_part, personPart)

            canvas[y1_dest:y2_dest, x1_dest:x2_dest] = combined
            print(f"  SUCCESS: Pasted person from box (x,y)={x},{y} at y_dest={y1_dest}.")

        except Exception as e:
            print(f"\n!!!! FAILED TO PASTE BOX (x,y)={x},{y}. REASON: {e} !!!!\n")

    return canvas


if __name__ == "__main__":
    #imagePath = "stock_group_photo.jpg"
    # imagePath = "img.png"
    #imagePath = "img_1.png"
    imagePath = "Images/imageProcessing.jpg"
    image = cv2.imread(imagePath)

    if image is None:
        print(f"No image found at {imagePath}")
    else:
        # Person Detection
        print("Finding people in Image using MediaPipe Face Detection...")
        mp_face_detection = mp.solutions.face_detection
        boxes = []

        # Initialize the Face Detection model
        with mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5) as face_detection:

            h_img, w_img = image.shape[:2]
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.detections:
                print("No faces found by MediaPipe Face Detection.")
            else:
                print(f"Found {len(results.detections)} faces.")
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x_face = int(bboxC.xmin * w_img)
                    y_face = int(bboxC.ymin * h_img)
                    w_face = int(bboxC.width * w_img)
                    h_face = int(bboxC.height * h_img)
                    padding_w = int(w_face * 1.0)
                    padding_h = int(h_face * 3.0)

                    HEAD_TOP_PADDING_FACTOR = 0.5
                    top_padding = int(h_face * HEAD_TOP_PADDING_FACTOR)

                    x = max(0, x_face - (padding_w // 2))
                    y = max(0, y_face - top_padding)
                    w = min(w_img - x, w_face + padding_w)
                    h = min(h_img - y, h_face + padding_h+top_padding)
                    boxes.append((x, y, w, h))

        if not boxes:
            print("No people found.")
        else:
            print(f"Generated {len(boxes)} bounding boxes from faces.")

            # --- NEW LOGIC: Find the alignment 'y' coordinate ---
            # Get all the 'y' (head) coordinates
            all_y_coords = [box[1] for box in boxes]
            # Find the head that is LOWEST on the screen (highest 'y' value)
            y_target_align = max(all_y_coords)
            print(f"Target head-level (lowest head) set to y={y_target_align}")

            # --- BUG 1 (INDENTATION) FIX ---
            # The 'with' block is now opened here and wraps EVERYTHING
            # that uses the segmenter.
            mp_selfie = mp.solutions.selfie_segmentation
            with mp_selfie.SelfieSegmentation(model_selection=1) as segmenter:

                master_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                print("Generating master mask for inpainting...")

                for box in boxes:
                    (x, y, w, h) = box
                    padding = 20
                    y1_roi = max(0, y - padding)
                    y2_roi = min(image.shape[0], y + h + padding)
                    x1_roi = max(0, x - padding)
                    x2_roi = min(image.shape[1], x + w + padding)

                    roi_original = image[y1_roi:y2_roi, x1_roi:x2_roi]
                    if roi_original.size == 0:
                        continue

                    rgb_roi = cv2.cvtColor(roi_original, cv2.COLOR_BGR2RGB)
                    results = segmenter.process(rgb_roi)

                    # --- BUG 2 (MODEL LIMITATION) FIX ---
                    # We must check if the segmenter actually returned a mask
                    if results.segmentation_mask is None:
                        print(f"Warning: Segmenter found no person in box {box}")
                        continue  # Skip this person

                    person_mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255

                    # Resize the mask to match the ROI
                    person_mask = cv2.resize(person_mask, (roi_original.shape[1], roi_original.shape[0]))

                    existing_mask_roi = master_mask[y1_roi:y2_roi, x1_roi:x2_roi]
                    combined_mask = cv2.bitwise_or(existing_mask_roi, person_mask)
                    master_mask[y1_roi:y2_roi, x1_roi:x2_roi] = combined_mask

                # Create inpainted background
                print("Inpainting background...")
                kernel = np.ones((10, 10), np.uint8)
                dilated_mask = cv2.dilate(master_mask, kernel, iterations=1)
                inpaintedBackground = cv2.inpaint(image, dilated_mask, 5, cv2.INPAINT_TELEA)

                # Call the main function
                # This is now SAFELY inside the 'with' block
                print("Compositing people onto background...")
                processed_canvas = segment_crop_and_align(image, boxes, inpaintedBackground, segmenter, y_target_align)

                # Show all results
                cv2.imshow("Original", image)
                cv2.imshow("Inpainted Background", inpaintedBackground)
                cv2.imshow("Processed Canvas", processed_canvas)

                print("Press any key to exit")
                cv2.waitKey(0)
                cv2.destroyAllWindows()