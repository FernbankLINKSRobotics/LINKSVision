import cv2
import numpy as np

largestArea = 0
cx, cy = 0, 0

#DEBUG Variables
debug = True #Do we want to see an image output?

filterArea = 1000

countDown = 0
photoCount = 0
outputPhotos = True

firstContour = None
secondContour = None

cap = cv2.VideoCapture(0)

while True:
    #Breaks the Camera Stream into Frames
    running, frame            = cap.read()
    
    if running:
        #First, we need to blur the image so contours are nicer
        frame = cv2.GaussianBlur(frame, (5,5), 0)
        
        #So, next we need to convert the frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #The Ranges of Green Color
        lower_green = np.array([20,50,50])
        upper_green = np.array([80,255,255])

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
        #mask = cv2.erode(mask, kernel, iterations = 1)
        #mask = cv2.dilate(mask, kernel, iterations = 1)

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

            #First off enumerate is a function that takes in an array and outputs an array of tuples with the first 
            #value being the index of the array and the second being the value from the array.
            #enumerate(map(lambda x: vc2.contoursArea(x),countours)) is a list of tuples where the value is equal to
            #the area of the contours area given earlier.
            #The rest of the line sortes that array in descending order from the value of the tuple
            areas = sorted(enumerate(map(lambda x: cv2.contoursArea(x), countours)), key=lambda x: x[1], reverse=True)
            
            #These find the indexes of the largest and second larges values of the contours and stores them.
            cnt1 = contours[areas[0][0]]
            cnt2 = contours[areas[1][0]]

            #Get the Moments of each contour
            M1 = cv2.moments(cnt1)
            M2 = cv2.moments(cnt2)

            if M1["m00"] != 0 and M2["m00"] != 0:
                #Average Center X
                cx1 = int(M1['m10']/M1['m00'])
                cx2 = int(M2['m10']/M2['m00'])
                cx = (cx1 + cx2) / 2

                #Average Center Y
                cy1 = int(M1['m01']/M1['m00'])
                cy2 = int(M2['m01']/M2['m00'])
                cy = (cy1 + cy2) / 2
                
        if debug:
            cv2.drawContours(res, contours, -1, (0, 255, 0), 3)
            cv2.circle(res,(cx, cy), 8, (0,0,255), -1)
            cv2.imshow("Initial", frame)
            cv2.imshow("Final", res)
            
    else: print("ERROR")

    if outputPhotos == False: continue

    countDown += 1
    if countDown >= 30:
        countDown = 0 
        cv2.imwrite("photos/photo_%i.jpg" % photoCount, res)
        photoCount += 1

    if cv2.waitKey(5) == ord("q"): break
cv2.destroyAllWindows()
