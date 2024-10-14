import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from google.generativeai import GenerativeModel, configure
import os
import streamlit as st
from PIL import Image


st.set_page_config(layout="wide")
st.image('mathGesture.png')
col1,col2=st.columns([2,1])

with col1:
    run=st.checkbox('Run',value=True)
    FRAME_WINDOW=st.image([])

with col2:
    output_text_area=st.title('Answer')
    output_text_area=st.subheader(" ")
configure(api_key='AIzaSyB13-ZEKNn76-teDHAwNA0gxyNiqKEyVdM')
model = GenerativeModel("gemini-1.5-flash")
# Initialize the webcam to capture video

# The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)


def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)

    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        lmList = hand1["lmList"]  # List of 21 landmarks for the first hand

        # Count the number of fingers up for the first hand
        fingers = detector.fingersUp(hand1)
        #print(f'list={fingers}')  # Print the count of fingers that are up
        return fingers,lmList
    else:
        return None

def draw(info,prev_pos,canvas):
    fingers,lmList=info
    curr_pos=None # every time we need to track by returning to none then assign new value.
    if fingers==[0,1,0,0,0]:# index finger is up then draw.
        curr_pos=lmList[8][0:2]# position of index finger(x,y).
        # lmLiat is a 2d list having every list containing position(x,y,z)
        # but we need x and y so slicing to get x and y.
        if prev_pos is None:
            prev_pos=curr_pos
        cv2.line(canvas,curr_pos,prev_pos,color=(225,0,225),thickness=10)
    elif fingers==[1,0,0,0,0]:
        canvas=np.zeros_like(img) # remove all
    return curr_pos,canvas

def sendToAI(model,canvas,fingers):
    if fingers==[1,1,1,1,0]:
        pil_image=Image.fromarray(canvas)#convert ndarray to pil image.
        response = model.generate_content(["Solve This math problem.",pil_image])
        #response = model.generate_content("Which is the Largest Ocean.")
        return response.text

prev_pos=None# previous position globally defined as none because there is nothing 1st.
canvas=None # draw on canvas
output_text=""
# Continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    success, img = cap.read()

    # Check if the frame was captured successfully
    if not success:
        st.error("Failed to capture image from webcam.")
        break

    img = cv2.flip(img, 1)  # flip the image horizontally
    if canvas is None:
        canvas = np.zeros_like(img)  # create a blank canvas

    info = getHandInfo(img)

    if info:
        fingers, lmList = info
        #print(fingers)
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text=sendToAI(model, canvas, fingers)

    image_combine = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combine, channels="BGR")
    if output_text:
        output_text_area.text(output_text)

    cv2.waitKey(1)
