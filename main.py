import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while True:
    #reads in the video and breaks it into frames
    _, frame            = cap.read()
    #blurs the images so that the result of the color filtering is better
    blur                = cv2.bilateralFilter(frame,9,75,75)
    #converts to HSV to filter the colors
    hsv                 = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    #sets the mininum and maximum colors for the filter
    lower_green         = np.array([30,100,100])
    upper_green         = np.array([50,255,255])
    #filters the colors
    mask                = cv2.inRange(hsv, lower_green, upper_green)
    res                 = cv2.bitwise_and(frame,frame, mask= mask)
    #converts the result to grayscale for the threshold and the contours (may remove this)
    gray                = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    #does a binary threshold of the grayscale (not being used now)
    _, thres            = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    #finds the contours of the grayscale
    contours, hierarchy = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #draws the contours to the screen
    cv2.drawContours(res,contours,-1,(0,255,0),3)
    #displays the video
    cv2.imshow('image',res)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
