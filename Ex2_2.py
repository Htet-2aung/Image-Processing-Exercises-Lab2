import cv2
import numpy as np

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Read an image from the webcam
ret, frame = cap.read()

if not ret:
    print("Error: Could not read frame.")
    exit()

# Release the webcam
cap.release()

# Increase the brightness of the image
brightness = 50
bright_img = cv2.convertScaleAbs(frame, beta=brightness)

# Enhance the image contrast using global histogram equalization
gray_img = cv2.cvtColor(bright_img, cv2.COLOR_BGR2GRAY)
equalized_img = cv2.equalizeHist(gray_img)

# Enhance the image contrast using adaptive histogram equalization (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray_img)

# Display the results
cv2.imshow('Original Image', frame)
cv2.imshow('Brightened Image', bright_img)
cv2.imshow('Global Histogram Equalization', equalized_img)
cv2.imshow('Adaptive Histogram Equalization (CLAHE)', clahe_img)

cv2.waitKey(0)
cv2.destroyAllWindows()