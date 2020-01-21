import cv2

img_path = '/workspace/Dataset/Img_wild/0000.jpg'

img = cv2.imread(img_path)
'''
img = img[:,82:942,:]

img = cv2.resize(img, (640,480))
'''
for i in range(1, 1000):
  cv2.imwrite('/workspace/Dataset/Img_wild/%04d.jpg'%i, img)

print('Done')