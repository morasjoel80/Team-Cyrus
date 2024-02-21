import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from playsound import playsound
import threading
import numpy as np
import math


MODEL_PATH = "Model/keras_model.h5"
LABEL_PATH = "Model/labels.txt"
SPEECH_PATH = "speech"
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2, detectionCon=0.8)
classifier = Classifier(MODEL_PATH, LABEL_PATH)

# Constants

OFFSET = 20
IMG_SIZE = 600

# Variables
wait = False
folder = open(LABEL_PATH, "r")
f = folder.read().splitlines()
Labels = f
print(Labels)
folder.close()


# Init text to speech

def speech(audio):
    global wait
    print(audio)
    if not wait:
        wait = True
        done = False
        while not done:
            try:
                playsound(f'{SPEECH_PATH}/{audio}.mp3')
                done = True
                wait = False
            except:
                continue


def capture():
    global wait
    text = ''
    while True:
        try:

            success, img = cap.read()
            imgOutput = img.copy()
            hands, img = detector.findHands(img)
            #   Finds Hands from img
            if hands:
                hand1 = hands[0]
                x, y, w, h = hand1['bbox']
                #   returns the Bounding Box attributes of hand

                imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
                #   Initialize white background to put cropped image onto

                if len(hands) == 1:     #   When one hand is detected
                    imgCrop = img[y - OFFSET: y + h + OFFSET, x - OFFSET: x + w + OFFSET]
                    #   Crops Hand with Bone structure

                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = IMG_SIZE / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((IMG_SIZE - wCal) / 2)
                        imgWhite[:, wGap: wCal + wGap] = imgResize
                        if not wait:
                            prediction, index = classifier.getPrediction(imgWhite)
                            print(prediction, index)
                        #   Pastes imgCrop onto imgWhite

                    else:
                        k = IMG_SIZE / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((IMG_SIZE - hCal) / 2)
                        imgWhite[hGap: hCal + hGap, :] = imgResize
                        if not wait:
                            prediction, index = classifier.getPrediction(imgWhite)
                        #   Pastes imgCrop onto imgWhite

                    # Text To Speech

                    if text != Labels[index]:
                        text = str(Labels[index])
                        threading.Thread(
                            target=speech, args=(text,)
                        ).start()
                    # Output Box

                    cv2.putText(imgOutput, Labels[index], (x, y - OFFSET), cv2.FONT_HERSHEY_COMPLEX, 2,
                                (255, 0, 255),
                                2)

                if len(hands) == 2:     #   When two hands are detected
                    hand2 = hands[1]
                    if hand1["type"] == "Right":    #   Checks whether the first hand to be detected is right
                        x, y, w, h = hand1['bbox']
                        centerpoint1 = hand1["center"]
                        x1, y1, w1, h1 = hand2['bbox']
                        centerpoint2 = hand2["center"]
                    else:
                        x, y, w, h = hand2['bbox']
                        centerpoint1 = hand2["center"]
                        x1, y1, w1, h1 = hand1['bbox']
                        centerpoint2 = hand1["center"]

                    length, info, img = detector.findDistance(centerpoint1, centerpoint2, img)

                    if y < y1:
                        #   Crops with respect to the left hand (if left hand is higher than the right)
                        imgCrop = img[y - OFFSET: info[3] +h1 + OFFSET, x - OFFSET: info[2] + w1 + (OFFSET + 50)]
                    else:
                        #   Crops with respect to the right hand (if right hand is higher than the left)
                        imgCrop = img[y1 - OFFSET: info[1] + h + OFFSET, x - OFFSET: info[2] + w1 + (OFFSET + 50)]

                    Havg = (info[1]+info[3])+(y+y1)/2
                    Wavg = (info[0]+info[2])+(x+x1)/2
                    aspectRatio = Havg / Wavg

                    if aspectRatio > 1:
                        k = IMG_SIZE / Havg
                        wCal = math.ceil(k * Wavg)
                        imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((IMG_SIZE - wCal) / 2)
                        imgWhite[:, wGap: wCal + wGap] = imgResize
                        if not wait:
                            prediction, index = classifier.getPrediction(imgWhite)
                            print(prediction, index)
                        #   Pastes imgCrop onto imgWhite

                    else:
                        k = IMG_SIZE / Wavg
                        hCal = math.ceil(k * Havg)
                        imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((IMG_SIZE - hCal) / 2)
                        imgWhite[hGap: hCal + hGap, :] = imgResize
                        if not wait:
                            prediction, index = classifier.getPrediction(imgWhite)
                        #   Pastes imgCrop onto imgWhite

                    # Text To Speech
                    if text != Labels[index] and not wait:
                        text = str(Labels[index])
                        threading.Thread(
                            target=speech, args=(text,)
                        ).start()
                    # Output Box

                    cv2.putText(imgOutput, Labels[index], (x, y - OFFSET), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255),
                                2)


                #   cv2.imshow("ImageCrop", imgCrop)
                #   Display cropped hand/s
                cv2.imshow("ImageWhite", imgWhite)
            cv2.imshow("Image", imgOutput)
            key = cv2.waitKey(1)

        except cv2.error:
            print("\nCannot Detect(Out of bounds)")
            # Prevents crash when hand is present outside the frame


if __name__ == "__main__":
    capture()