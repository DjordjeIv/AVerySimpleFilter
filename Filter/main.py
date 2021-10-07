import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

ImgIn = cv.imread("Grudvice.png")

ImgBlur = cv.medianBlur(ImgIn, 11)

ImgHSV = cv.cvtColor(ImgBlur, cv.COLOR_BGR2HSV)

ImgSat = ImgHSV[:, :, 0]
#
if True:
   plt.imshow(ImgSat)
   plt.show()
#
ImgTh = cv.inRange(ImgSat, 30, 100)
cv.imshow("Threshold", ImgTh)
#
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
ImgOpen = cv.morphologyEx(src=ImgTh, op=cv.MORPH_OPEN, kernel=kernel)
cv.imshow("Open", ImgOpen)
#
imgOut = ImgIn.copy()
cntCC, imgCC = cv.connectedComponents(ImgOpen, connectivity=4)
#
maxCnt = 0
maxBBox = None
for cc in range(1, cntCC):
    imgCurr = np.where(imgCC == cc, 255, 0).astype(np.uint8)
    x, y, w, h = cv.boundingRect(imgCurr)
    cnt = imgCurr.sum() / 255
    if cnt > maxCnt:
        maxCnt = cnt
        maxBBox = x, y, w, h
    cv.rectangle(imgOut, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)

x, y, w, h = maxBBox
cv.rectangle(imgOut, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

cv.putText(imgOut, text='CNT: ' + str(cntCC), org=(5, 17), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
           color=(0, 0, 255), thickness=2)
cv.imshow("Output", imgOut)
cv.imshow("Blur", ImgBlur)

cv.imshow("Hue", ImgHSV[:, :, 0])

cv.imshow("Input", ImgIn)

cv.imwrite("Input.png", ImgIn)
cv.imwrite("Range.png", ImgTh)
cv.imwrite("Output.png", imgOut)

cv.waitKey(0)
