import cv2 as cv
import numpy as np

# Load the image
img = cv.imread('Lab02\lab02_ex.png')

# Check if the image was loaded successfully
if img is None:
    raise FileNotFoundError("The image file 'lab02_ex.png' was not found or could not be loaded.")

# 1. Split each color channel of the image
b, g, r = cv.split(img)

# 2. Locate the position of each balloon by drawing a rectangle (bounding-box) surrounding each balloon
# Assuming we have predefined positions for the balloons
balloon_positions = {
    'red': (50, 50, 150, 150),
    'green': (200, 50, 300, 150),
    'blue': (350, 50, 450, 150),
    'yellow': (500, 50, 600, 150)
}

for color, (x1, y1, x2, y2) in balloon_positions.items():
    cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv.putText(img, color, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# 4. Extract the yellow balloon by creating a new image of only one balloon
x1, y1, x2, y2 = balloon_positions['yellow']
yellow_balloon = img[y1:y2, x1:x2]
cv.imwrite('yellow_balloon.jpg', yellow_balloon)

# 5. Extract the yellow balloon automatically by using HSV color space to extract only pixels of yellow color
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
mask = cv.inRange(hsv, lower_yellow, upper_yellow)
yellow_extracted = cv.bitwise_and(img, img, mask=mask)
cv.imwrite('yellow_extracted.jpg', yellow_extracted)

# 6. Re-paint the yellow balloon by replacing the pixels of yellow by green
img[mask > 0] = [0, 255, 0]
cv.imwrite('yellow_to_green.jpg', img)

# 7. Rotate the first balloon an angle of 20 degrees
x1, y1, x2, y2 = balloon_positions['red']
red_balloon = img[y1:y2, x1:x2]
(h, w) = red_balloon.shape[:2]
center = (w // 2, h // 2)
M = cv.getRotationMatrix2D(center, 20, 1.0)
rotated_red_balloon = cv.warpAffine(red_balloon, M, (w, h))
img[y1:y2, x1:x2] = rotated_red_balloon
cv.imwrite('rotated_red_balloon.jpg', img)

# Display the final image with all modifications
cv.imshow('Balloons', img)
cv.waitKey(0)
cv.destroyAllWindows()