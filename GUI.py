import cv2
import cvzone
import numpy as np
from tkinter import *
from PIL import Image, ImageTk

window = Tk()
window.geometry('900x720')
window.configure(bg='black')
window.title("Sign Language")

F1 = LabelFrame(window, bg='grey')
F1.pack()
L1 = Label(F1, bg='grey')
L1.pack()

cap = cv2.VideoCapture(0)

while True:
    img = cap.read()[1]
    img2 = img.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = ImageTk.PhotoImage(Image.fromarray(img2))
    L1["image"] = img2
    window.update()