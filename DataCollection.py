import cv2
from cvzone.HandTrackingModule import HandDetector
cap = cv2.VideoCapture(0)
detector = HandDetector()

while True:
    success,img= cap.read()
    hands,img = detector.findHands(img)
    cv2.imshow("image",img)
    cv2.waitKey(1)


