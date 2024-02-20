import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
#   Initialization
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2, detectionCon=0.8)


#   Constants
OFFSET = 20
IMG_SIZE = 600

#   Variables
counter = 0
folder = "Data/Thank you"

while True:
    try:
        success, img = cap.read()
        hands, img = detector.findHands(img)

        if hands:
            hand1 = hands[0]
            x, y, w, h = hand1["bbox"]

            imgwhite = np.ones([IMG_SIZE, IMG_SIZE, 3], np.uint8) * 255
            #   White Background
            imgcrop = img[y - OFFSET: y + h + OFFSET, x - OFFSET: x + w + OFFSET]

            if len(hands) == 1:

                aspect_ratio = h/w

                if aspect_ratio > 1:
                    k = IMG_SIZE / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgcrop, (wCal, IMG_SIZE))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((IMG_SIZE - wCal) / 2)
                    imgwhite[:, wGap: wCal + wGap] = imgResize

                else:
                    k = IMG_SIZE / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgcrop, (IMG_SIZE, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((IMG_SIZE - hCal) / 2)
                    imgwhite[hGap: hCal + hGap, :] = imgResize

                cv2.imshow("Crop", imgcrop)
                cv2.imshow("White", imgwhite)

            if len(hands) == 2:
                hand2 = hands[1]
                if hand1["type"] == "Right":
                    #   If first hand to be detected is right
                    x, y, w, h = hand1["bbox"]
                    centerpoint1 = hand1["center"]
                    x1, y1, w1, h1 = hand2["bbox"]
                    centerpoint2 = hand2["center"]

                else:
                    #   If first hand to be detected is left
                    x, y, w, h = hand2["bbox"]
                    centerpoint1 = hand2["center"]
                    x1, y1, w1, h1 = hand1["bbox"]
                    centerpoint2 = hand1["center"]

                length, info, img = detector.findDistance(centerpoint1, centerpoint2, img)

                if y < y1:
                    #   Crops with respect to the left hand (if left hand is higher than the right)
                    imgCrop = img[y - OFFSET - 50: info[3] + h1 + OFFSET, x - OFFSET: info[2] + w1 + (OFFSET + 50)]
                else:
                    #   Crops with respect to the right hand (if right hand is higher than the left)
                    imgCrop = img[y1 - OFFSET - 50: info[1] + h + OFFSET, x - OFFSET: info[2] + w1 + (OFFSET + 50)]

                Havg = (info[1] + info[3]) + (y + y1) / 2
                Wavg = (info[0] + info[2]) + (x + x1) / 2
                aspectRatio = Havg / Wavg

                if aspectRatio > 1:
                    k = IMG_SIZE / Havg
                    wCal = math.ceil(k * Wavg)
                    imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((IMG_SIZE - wCal) / 2)
                    imgwhite[:, wGap: wCal + wGap] = imgResize

                else:
                    k = IMG_SIZE / Wavg
                    hCal = math.ceil(k * Havg)
                    imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((IMG_SIZE - hCal) / 2)
                    imgwhite[hGap: hCal + hGap, :] = imgResize

                cv2.imshow("Crop", imgcrop)
                cv2.imshow("White", imgwhite)

        cv2.imshow("image", img)
        key = cv2.waitKey(1)

        if key == ord("s"):
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgwhite)
            #   Saves samples in the specified folder
            print(counter)

    except cv2.error:
        print("\n Cannot Detect (Out of Bounds)")




