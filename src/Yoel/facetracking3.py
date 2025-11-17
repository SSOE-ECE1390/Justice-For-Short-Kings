import cv2
import mediapipe as mp

# Mediapipe objects
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# Load the image
img = cv2.imread("people.jpg")  # <-- replace with your image
h, w, _ = img.shape

# Convert to RGB for MediaPipe
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create the face detector
with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    results = face_detection.process(img_rgb)

    shortest_box = None      # (x, y, w_box, h_box)
    shortest_height = None   # store min bounding box height

    if results.detections:
        for detection in results.detections:
            # Draw normal detection annotation (green box + keypoints)
            mp_draw.draw_detection(img, detection)

            # Extract bounding box (relative coords -> pixels)
            relative_box = detection.location_data.relative_bounding_box
            x = int(relative_box.xmin * w)
            y = int(relative_box.ymin * h)
            w_box = int(relative_box.width * w)
            h_box = int(relative_box.height * h)

            # Draw our own rectangle (optional, for clarity)
            cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

            # Update shortest person (smallest face height)
            if shortest_height is None or h_box < shortest_height:
                shortest_height = h_box
                shortest_box = (x, y, w_box, h_box)

    # Highlight the shortest person
    if shortest_box is not None:
        x, y, w_box, h_box = shortest_box

        # Draw a thicker, different-colored rectangle (e.g., red)
        cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (0, 0, 255), 3)

        
# Show image with faces and shortest person highlighted
cv2.imshow("Detected Faces with Shortest Person", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
