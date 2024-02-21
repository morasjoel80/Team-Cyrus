import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from playsound import playsound
import threading
import numpy as np
import math
from PIL import Image
from toga import App, Box, Button, ImageView, Label, SplitContainer, Icon

#   OpenCV

MODEL_PATH = "Model/keras_model.h5"
LABEL_PATH = "Model/labels.txt"
SPEECH_PATH = "speech"
cap = None
detector = HandDetector(maxHands=2, detectionCon=0.8)
classifier = Classifier(MODEL_PATH, LABEL_PATH)
# Constants

OFFSET = 20
IMG_SIZE = 600

# Variables
cam_on = False
wait = False
folder = open(LABEL_PATH, "r")
f = folder.read().splitlines()
Labels = f
print(Labels)
folder.close()

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

class SignLanguageApp(App):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cam_on = False
        self.cap = None
        self.labels = []
        self.wait = False

    def on_cam(self, widget):
        self.cap = cv2.VideoCapture(0)
        self.cam_on = True

    def off_cam(self, widget):
        self.cap.release()
        self.cam_on = False

    def build(self):
        main_box = Box()

        # Create Toga widgets
        button_start = Button('Start', on_press=self.on_cam)
        button_stop = Button('Stop', on_press=self.off_cam)
        image_view = ImageView(style=ImageView.Style.FIT)

        # Add widgets to the main box
        main_box.add(button_start)
        main_box.add(button_stop)
        main_box.add(image_view)

        # Set up the main content
        main_content = SplitContainer()
        main_content.content = [main_box]

        # Set up the main window
        self.main_window = main_content

        return self.main_window

    def update_image(self, img):
        # Convert the image to a format suitable for Toga ImageView
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.main_window.content[0].content[2].value = img


def capture(app):
    text = ''
    while True:
        if app.cam_on:
            try:
                success, img = app.cap.read()
                imgOutput = img.copy()

                # Your existing image processing code goes here
                hands, img = detector.findHands(img)
                #   Finds Hands from img
                if hands:
                    hand1 = hands[0]
                    x, y, w, h = hand1['bbox']
                    #   returns the Bounding Box attributes of hand

                    imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
                    #   Initialize white background to put cropped image onto

                    if len(hands) == 1:  # When one hand is detected
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

                    if len(hands) == 2:  # When two hands are detected
                        hand2 = hands[1]
                        if hand1["type"] == "Right":  # Checks whether the first hand to be detected is right
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
                            imgCrop = img[y - OFFSET: info[3] + h1 + OFFSET, x - OFFSET: info[2] + w1 + (OFFSET + 50)]
                        else:
                            #   Crops with respect to the right hand (if right hand is higher than the left)
                            imgCrop = img[y1 - OFFSET: info[1] + h + OFFSET, x - OFFSET: info[2] + w1 + (OFFSET + 50)]

                        Havg = (info[1] + info[3]) + (y + y1) / 2
                        Wavg = (info[0] + info[2]) + (x + x1) / 2
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

                        cv2.putText(imgOutput, Labels[index], (x, y - OFFSET), cv2.FONT_HERSHEY_COMPLEX, 2,
                                    (255, 0, 255),
                                    2)
                # Update the Toga ImageView
                app.update_image(imgOutput)
            except cv2.error:
                print("\nCannot Detect(Out of bounds)")
        else:
            # Display a black image when the camera is off
            img_black = np.ones((500, 500, 3), np.uint8)
            img_black = cv2.cvtColor(img_black, cv2.COLOR_BGR2RGB)
            app.update_image(img_black)


if __name__ == "__main__":
    app = SignLanguageApp('Sign Language', 'org.example.signlanguageapp', startup=capture)
    app.main_loop()
