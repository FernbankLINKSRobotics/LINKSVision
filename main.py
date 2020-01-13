import cv2
import math
import subprocess
import numpy as np
from networktables import NetworkTables

NetworkTables.initialize(server="roborio-4468-frc.local")
table = NetworkTables.getTable("vision")


largestArea = 0
cx, cy = 0, 0

# DEBUG Variables
debug = False  # Do we want to see an image output print and send?
showing = False  # Will show the processed image
printing = True  # Will print the processed values
sending = False  # Will send the values to the RIO hopefully

cap = cv2.VideoCapture(0)

# The following vars and functions are for distance calcs
# Raw data
fov = 62.8  # Camera field of view
bh = 2.5  # Height of the boiler
ch = 0.5  # Height of the camera
Ao = 50  # Angle offset or camera angle
Iw = 640  # Image width
Ih = 360  # Image height

# Interpretted and calculated vals
deltaH = bh - ch  # difference in heights
Iwc = (Iw / 2) - 0.5  # The center pixel for width
Ihc = (Ih / 2) - 0.5  # The center pixel for height
f = Iw / (2 * math.tan(fov / 2))  # The focal length of the camera from the FOV

# Calculates the horizontal angle from the boiler
def yawAngle(x):
    return math.atan((x - Iwc) / f)


# Calculates the distance to the boiler
def distance(y):
    return deltaH / math.tan(math.atan((y - Ihc) / f) + Ao)


if debug:
    printing, showing = True, True

while True:
    # Breaks the Camera Stream into Frames
    running, frame = cap.read()

    if running:
        table.putBoolean("Online?", True)
        # First, we need to down res the image for bluring as well as speed
        frame = cv2.resize(frame, (Iw, Ih), interpolation=cv2.INTER_CUBIC)

        # So, next we need to convert the frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # The Ranges of Green Color
        lower_green = np.array([20, 50, 50])
        upper_green = np.array([80, 255, 255])
        # NOTE the values i got in in c++ where
        # [20,100,50] and [100,190,140] but i may have over fit

        # Now, we need to find out what is in that range
        # OpenCV will return a binary image where white = 1
        # and black = 0
        mask = cv2.inRange(hsv, lower_green, upper_green)

        if debug:
            # This produces an image that is a combination
            # of the mask and original frame
            res = cv2.bitwise_and(hsv, frame, mask=mask)

        # Now, let's find some contours!
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Needed otherwise the program will crash bc we have to deal with 2 contours
        if len(contours) >= 2:
            largestArea = 0
            secondLargestArea = 0

            # So, this is an array of all of the contour
            # areas, and they are added to this array
            # from the main contour array. This is done
            # so we can find the largest and second largest
            # contours as those will be the tape

            # First off enumerate is a function that takes in an array
            # and outputs an array of tuples with the first
            # value being the index of the array and the second being the value from the array.
            # enumerate(map(lambda x: vc2.contoursArea(x),countours)) is a
            # list of tuples where the value is equal to
            # the area of the contours area given earlier.
            # The rest of the line sortes that array in descending order from the value of the tuple
            areas = sorted(
                enumerate(map(lambda x: cv2.contourArea(x), contours)),
                key=lambda x: x[1],
                reverse=True,
            )

            # These find the indexes of the largest and second larges
            # values of the contours and stores them.
            cnt1 = contours[areas[0][0]]
            cnt2 = contours[areas[1][0]]

            # Get the Moments of each contour
            M1 = cv2.moments(cnt1)
            M2 = cv2.moments(cnt2)

            if M1["m00"] != 0 and M2["m00"] != 0:
                # Average Center X
                # Finds the center of mass for X of the hulls
                cx1 = int(M1["m10"] / M1["m00"])
                cx2 = int(M2["m10"] / M2["m00"])
                cx = (cx1 + cx2) / 2

                # Average Center Y
                # Finds the center of mass for Y of the hulls
                cy1 = int(M1["m01"] / M1["m00"])
                cy2 = int(M2["m01"] / M2["m00"])
                cy = (cy1 + cy2) / 2

                # Sends the angle and distance to the RIO and if true
                if sending:
                    table.putNumber("Angle", yawAngle(cx))
                    table.putNumber("Distance", distance(cy))

        if showing:
            # Shows the contours on the screen
            cv2.drawContours(res, contours, -1, (255, 199, 0), 2)
            # Draws a circle on the average values
            cv2.circle(res, (cx, cy), 10, (0, 0, 255), -1)
            # Displays the before and after frames
            cv2.imshow("Initial", frame)
            cv2.imshow("Final", res)

    else:
        print("ERROR")
        table.putNumber("Angle", 0)
        table.putNumber("Distance", 0)

    # Shutsdown
    if cv2.waitKey(5) == ord("q") or table.getBoolean("Shutdown", False):
        break
