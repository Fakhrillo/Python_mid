import cv2
import numpy as np
import time
import os
import Handtracking as htm


brushThickness = 15
eraserThickness = 50

folderPath = "Header"
mylist = os.listdir(folderPath)
overlayList = []

for imPath in mylist:
    image = cv2.imread(f"{folderPath}/{imPath}")
    overlayList.append(image)

header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.75)
yp, xp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPoints(img)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

         # 3. Check which fingers up
        fingers = detector.fingersUp()

         # 4. If selection mode - Two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            # print("selection mode")
            if y1 < 125:
                if 250<x1<450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550<x1<750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800<x1<900:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1000<x1<1150:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (x1, y1-25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. If drawing mode - One finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            #print("drawing mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp),(x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp),(x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp),(x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp),(x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    img[0:125, 0:1280] = header
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)