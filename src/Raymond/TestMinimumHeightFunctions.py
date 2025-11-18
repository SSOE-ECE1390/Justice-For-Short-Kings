import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from MinimumHeightFunctions import (
    cropToShortestPersonAndInpaint,
    scaleToShortestPersonAndPaste,
)
def runMinHeightFunctionsSelector():
    INPUT_IMAGE_PATH = "Images/image2.png"
    BACKGROUND_IMAGE_PATH = "Background_Images/background_photo2.png"

    print("\nMINIMUM HEIGHT FUNCTIONS SELECTOR")
    choice = input ("Enter 1 for Crop To Shortest, or 2 for Scale To Shortest: ")
    print("-"*40)
    try:
        if choice == '1':
            print("🚀 Running: Alignment Only (Inpainting) method...")
            cropToShortestPersonAndInpaint(INPUT_IMAGE_PATH)

        elif choice == '2':
            print("🔬 Running: Scaling and Equalization (Final) method...")
            scaleToShortestPersonAndPaste(INPUT_IMAGE_PATH, BACKGROUND_IMAGE_PATH)

        else:
            print("Invalid choice. Please enter '1' or '2'.")

    except Exception as e:
        print(f"\nAN EXECUTION ERROR OCCURRED: {e}")
        print("Ensure all MediaPipe models are initialized and inputs are valid.")

if __name__ == '__main__':
    runMinHeightFunctionsSelector()