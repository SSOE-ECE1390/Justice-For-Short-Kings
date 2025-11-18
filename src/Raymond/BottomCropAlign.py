import cv2
import numpy as np

def bottom_crop_and_align(original_image, bounding_box):
    """
    Equalizes height by cropping taller people and aligning all heads
    :param original_image: The source image
    :param bounding_box: A list of tuples in the format of (x,y,w,h)
    :return: np.array A new image with the processed people.
    """

    if not bounding_box:
        print("Warning: No bounding box provided")
        return original_image

    # Find the Shortest Person
    minHeight = float('inf')
    reference_box = None

    for box in bounding_box:
        (x,y,w,h) = box
        if h < minHeight:
            minHeight = h
            reference_box = box

    # Get the target values
    (ref_x,ref_y,ref_w,ref_h) = reference_box
    height_target = ref_h
    y_target = ref_y

    print(f"Reference Found: Target Height: {height_target}px, Target Head Leve: y: {y_target}px")

    # Create New Image canvas
    canvas = np.zeros_like(original_image)

    for box in bounding_box:
        (x,y,w,h) = box

        # Define Region Of Interests
        y1_source = y
        y2_source = y + height_target
        x1_source = x
        x2_source = x + w

        # Extract the Patch
        source_roi = original_image[y1_source:y2_source, x1_source:x2_source]

        # Get the actual shape of the image
        actual_h, actual_w, _ = source_roi.shape

        # Define Destination Region of Interests
        y1_dest = y_target
        y2_dest = y_target + actual_h
        x1_dest = x
        x2_dest = x + actual_w

        # Perform the paste
        try:
            canvas[y1_dest:y2_dest, x1_dest:x2_dest] = source_roi
        except ValueError as e:
            print(f"Error pasting box at (x,y) = {x},{y}. Shape mismatch: {e}")
            pass

    return canvas

if __name__ == "__main__":
    dummyImage = np.zeros((500,500,3), np.uint8)
    boxes = [
        (50,200,80,150),
        (200,100,80,300),
        (350, 150,80,200)
    ]

    cv2.rectangle(dummyImage, (50,200), (50+80, 200+150), (0, 255, 0), 2)
    cv2.rectangle(dummyImage, (200,100), (200+80, 100+300), (0,0,255),2)
    cv2.rectangle(dummyImage, (350, 150), (350 + 80, 150 + 200), (255, 0, 0), 2)

    processed_canvas = bottom_crop_and_align(dummyImage, boxes)

    cv2.imshow("Original Image", dummyImage)
    cv2.imshow("Processed Image", processed_canvas)

    print("Showing images. Press any key to exit.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()