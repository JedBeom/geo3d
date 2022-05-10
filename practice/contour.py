import cv2
import numpy as np

src = cv2.imread("g2.png", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(src, 100, 255)
imgLines = cv2.HoughLinesP(canny, 15, np.pi / 180, 10, minLineLength=10, maxLineGap=30)


for i in range(len(imgLines)):
    for x1, y1, x2, y2 in imgLines[i]:
        cv2.line(src, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('canny', canny)
cv2.imshow('Final Image with dotted Lines detected', src)
cv2.waitKey()
cv2.destroyAllWindows()

'''
_, binary = cv2.threshold(canny, 127, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_not(binary)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    cv2.drawContours(src, [contours[i]], 0, (0, 0, 225), 2)
    cv2.putText(src, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
    print(i, hierarchy[0][i])
    cv2.imshow("src", src)
    cv2.waitKey()

cv2.destroyAllWindows()
'''
