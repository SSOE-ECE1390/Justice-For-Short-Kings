import cv2
import mediapipe as mp
from math import sqrt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    counter = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                nose = face_landmarks.landmark[1]
                chin = face_landmarks.landmark[152]
                dist = sqrt((nose.x - chin.x)**2 + (nose.y - chin.y)**2)

                if counter % 1000 == 0:
                    print(f"Face distance (nose to chin): {dist}")
                    print(f"Nose coordinates: X={nose.x}, Y={nose.y}")
                counter += 1

                # Draw face mesh (tesselation)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                # Draw contours (eyes, lips, face outline)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

        cv2.imshow('MediaPipe Face Tracking', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()


