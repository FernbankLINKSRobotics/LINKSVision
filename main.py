import cv2
import numpy as np

largestArea = 0
cx, cy = 0, 0

#DEBUG Variables
debug = True #Do we want to see an image output?

filterArea = 1000

firstContour = None
secondContour = None

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_EXPOSURE, 1)

while True:
    #Breaks the Camera Stream into Frames
    running, frame            = cap.read()

    if running:
        #First, we need to blur the image so contours are nicer
        frame = cv2.GaussianBlur(frame, (5,5), 0)
        
        #So, next we need to convert the frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #The Ranges of Green Color
        lower_green = np.array([80,100,100])
        upper_green = np.array([100,255,255])

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
        if len(contours) >= 2:
            largestArea = 0
            secondLargestArea = 0

            #So, this is an array of all of the contour
            #areas, and they are added to this array
            #from the main contour array. This is done
            #so we can find the largest and second largest
            #contours as those will be the tape
            areas = []
            for cnt in contours:
                areas.append(cv2.contourArea(cnt))

            for area in areas:
                if area > largestArea:
                    #Now, we find the largest area
                    largestArea = area

            #and get the index of it from the array
            indexOfLargest = areas.index(area)
            cnt1 = contours[indexOfLargest]
            
            for area in areas:
                #So, finding the second largest is a little
                #tricky. What I think we should do is iterate
                #through the array again, but if we are at the
                #largest area, we skip over it.
                if area == areas[indexOfLargest]:
                    continue
                #and keep going until we find the second largest
                if area > secondLargestArea:
                    secondLargestArea = area
            
            indexOfSecondLargest = areas.index(area)
            cnt2 = contours[indexOfSecondLargest]

            #Get the Moments of each contour
            M1 = cv2.moments(cnt1)
            M2 = cv2.moments(cnt2)

            if M1["m00"] != 0 and M2["m00"] != 0:
                #Average Center X
                cx1 = int(M1['m10']/M1['m00'])
                cx2 = int(M2['m10']/M2['m00'])
                cx = (cx1 + cx2) / 2.0

                #Average Center Y
                cy1 = int(M1['m01']/M1['m00'])
                cy2 = int(M2['m01']/M2['m00'])
                cy = (cy1 + cy2) / 2.0
                
        if debug:
            cv2.drawContours(res, contours, -1, (0, 255, 0), 3)
            cv2.circle(res,(cx, cy), 8, (0,0,255), -1)
            cv2.imshow("Initial", frame)
            cv2.imshow("Final", res)
            
    else:
        print("ERROR")

    if cv2.waitKey(5) == ord("q"): break
cv2.destroyAllWindows()
