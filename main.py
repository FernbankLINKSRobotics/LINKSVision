import cv2
import numpy as np
from networktables import NetworkTable


debug = True
largestArea = 0
cx, cy = 0, 0

#Set up NetworkTables

#This is the actual video stream
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_EXPOSURE, -1.0)
while True:
    #reads in the video and breaks it into frames
    running, frame            = cap.read()

    if(running):
        #First, we need to blur the image so contours are nicer
        frame = cv2.GaussianBlur(frame, (5,5), 0)
        
        #So, next we need to convert the frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #The Ranges of Green Color
        lower_green = np.array([70,50,50])
        upper_green = np.array([90,255,255])

        #Now, we need to find out what is in that range
        #OpenCV will return a binary image where white = 1
        #and black = 0
        mask = cv2.inRange(hsv, lower_green, upper_green)

        #The Kernel is just an array that is scanned over the
        #image. It is full of ones so it doesn't change anything.
        kernel = np.ones((5,5), np.uint8)

        #Now, we need to remove more noise! This can be done
        #with erosion. When we erode, the program looks to see
        #if a zero is in the neighborhood of its scan. If it is,
        #the range of that kernel is 0.
        mask = cv2.erode(mask, kernel, iterations = 1)
        mask = cv2.dilate(mask, kernel, iterations = 1)

        if debug:
            #This produces an image that is a combination
            #of the mask and original frame
            res = cv2.bitwise_and(hsv, frame, mask = mask)

        #Now, let's find some contours!
        _, contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        #Needed otherwise the program will crash
        if len(contours) >= 1:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= largestArea :
                    largestArea = area
                    M = cv2.moments(cnt)
                    #Also needed unless you like errors
                    if M["m00"] != 0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        print("Center X: %s" %(cx))
                        print("Center Y: %s" %(cy))
                    
            

        if debug:
            cv2.drawContours(res, contours, -1, (0, 255, 0), 3)
            cv2.imshow("Initial", frame)
            cv2.imshow("Final", res)
        if cv2.waitKey(5) & 0xFF == 27: break
