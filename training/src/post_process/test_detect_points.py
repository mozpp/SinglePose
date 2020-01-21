import cv2
import numpy as np
import matplotlib.pyplot as plt

# img_path = '../../demo_io/limb_hm/22.png'
img_path = '../../demo_io/img_and_heat_conn/17.png'

img = cv2.imread(img_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 200)

cv2.imshow('test', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

lines = cv2.HoughLinesP(edges, 1, np.pi/180,threshold=30,minLineLength=30, maxLineGap=10)
ln = lines[:,0,:]
print(ln)

for x1, y1, x2, y2 in ln:
    cv2.line(img, (x1,y1),(x2,y2),(0,0,255),1)

cv2.imshow('test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()