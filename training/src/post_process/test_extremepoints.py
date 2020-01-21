# import the necessary packages
import imutils
import cv2
import numpy as np

def show(img):
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_img_with_contour(image, contour):
    img = image.copy()
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for point in contour[0]:
        point = point[0]
        print('==>', point)
        cv2.line(img, tuple(point), tuple(point), (0,0,255), thickness=3)
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_img_with_extreme_points(image, points):
    img = image.copy()
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for point in points:
        cv2.line(img, tuple(point), tuple(point), (0,0,255), thickness=3)
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    d = (x1 - x2)**2 + (y1 - y2)**2
    return np.sqrt(1.0 * d)

def mid_point(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return int((x1 + x2)*0.5), int((y1 + y2)*0.5)


def get_end_points(extreme_points):
    position = {}
    distances = []
    count = 0
    for i in range(len(extreme_points) - 1):
        for j in range(i+1, len(extreme_points)):
            d = distance(extreme_points[i], extreme_points[j])
            distances.append(d)
            position[count] = (i, j)
            count += 1

    loc1 = np.argmin(distances)
    distances[loc1] = np.inf
    loc2 = np.argmin(distances)

    end1 = mid_point(extreme_points[position[loc1][0]], extreme_points[position[loc1][1]])
    end2 = mid_point(extreme_points[position[loc2][0]], extreme_points[position[loc2][1]])

    return end1, end2

if __name__ == '__main__':

    for i in range(14, 26):


        # load the image, convert it to grayscale, and blur it slightly
        # img_path = '../../demo_io/img_and_heat_conn/21.png'
        # img_path = '../../demo_io/limb_hm/29.png'
        img_path = '../../demo_io/img_and_heat_conn/{:d}.png'.format(i)

        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        # show(gray)

        # find contours in thresholded image, then grab the largest
        # one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(cnts)
        # print('==>',cnts[0][0])
        # show_img_with_contour(gray, cnts)

        cnts = imutils.grab_contours(cnts)
        # print(cnts)
        # show_img_with_contour(gray, cnts)
        #
        # print(cnts)

        c = max(cnts, key=cv2.contourArea)
        # show_img_with_contour(gray, cnts)

        print('c==>',c, c.shape)


        # determine the most extreme points along the contour
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])


        show_img_with_extreme_points(gray, [extLeft, extRight, extTop, extBot])

        end1, end2 = get_end_points([extLeft, extRight, extTop, extBot])

        show_img_with_extreme_points(gray, [end1, end2])


