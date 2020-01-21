import numpy as np
import numpy.matlib as mt
import cv2
import math
import time
import matplotlib.pyplot as plt
from PIL import Image


img = plt.imread('/home/yangfeiyu/Downloads/test_1_rk.jpg')
plt.figure()
plt.imshow(img)

theta = 45
plt.figure()
img = np.array(Image.fromarray(img).rotate(theta))
plt.imshow(img)

plt.show()