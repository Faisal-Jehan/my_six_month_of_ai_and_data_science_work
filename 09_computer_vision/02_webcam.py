import numpy as np
import cv2 as cv
# read webcam using cv
cap = cv.VideoCapture(1)
if cap.isOpened():
    ret, frame = cap.read()
else:
    ret = False
while ret:
    ret, frame = cap.read()
    cv.imshow("webcam", frame)
    if cv.waitkey(25) & 0xFF ==ord("q"):
        break
cap.release()
cv.destroyAllWindows()