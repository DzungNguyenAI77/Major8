import numpy as np
import cv2
import mss
import time

# Define the screen area to capture (adjust for your game window)
monitor = {"top": 100, "left": 0, "width": 800, "height": 600}

# Initialize MSS for screen capture
sct = mss.mss()


def capture_screen():
    # Capture screen and convert to numpy array
    screen = np.array(sct.grab(monitor))

    # Convert to grayscale (helps reduce noise and unnecessary information)
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    # Optional: You can apply edge detection or thresholding here
    edges = cv2.Canny(gray_screen, threshold1=100, threshold2=200)

    return edges


# Loop to continuously capture screen (for testing)
while True:
    frame = capture_screen()
    cv2.imshow("Screen", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
