import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

#   Initialization
cap = cv2.VideoCapture(0)
detector = HandDetector()


#   Constants
OFFSET = 20
IMG_SIZE = 600

while True:
    success,img= cap.read()
    hands,img = detector.findHands(img)

    if hands:
        hand1 = hands[0]
        x, y, w, h = hand1["bbox"]

        imgwhite = np.ones([IMG_SIZE, IMG_SIZE, 3], np.uint8) * 255
        #   White Background
        imgcrop = img[y - OFFSET: y + h + OFFSET, x - OFFSET: x + w + OFFSET]

        aspect_ratio = h/w
        cv2.imshow("Crop", imgcrop)
        cv2.imshow("White", imgwhite)
    cv2.imshow("image", img)
    cv2.waitKey(1)


