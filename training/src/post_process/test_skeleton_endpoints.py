import cv2
import numpy as np

def skeleton_endpoints(skel):
    skel = skel.copy()
    skel[skel != 0] = 1
    skel = np.uint8(skel)

    kernel = np.uint8([[1,1, 1],
                       [1,10,1],
                       [1,1, 1],])
    src_depth = -1
    filtered = cv2.filter2D(skel, src_depth, kernel)

    out = np.zeros_like(skel)
    print(np.where(filtered == 11))
    out[np.where(filtered == 11)] = 1
    return out


def show(img):
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    img_path = '../../demo_io/img_and_heat_conn/21.png'

    img = cv2.imread(img_path)
    # show(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show(gray)


    endpoints = skeleton_endpoints(gray)
    show(endpoints)

