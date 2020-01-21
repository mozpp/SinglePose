import cv2
import numpy as np

def fuse_image_with_hm(org_img, org_hm, alpha):
    '''
    show img with hms as a mask
    '''
    img = org_img.copy()
    hm = org_hm.copy()

    h, w = hm.shape
    img = cv2.resize(img, (w, h))

    hm = hm / np.max(hm) * 255
    hm = hm.astype(np.uint8)
    hm = np.stack(3*[hm], axis=-1)
    img_with_hm = alpha * img + (1 - alpha) * hm
    img_with_hm = img_with_hm.astype(np.uint8)

    # transpose if used by cv2
    img_with_hm = img_with_hm[:,:,::-1]
    return img_with_hm

if __name__ == '__main__':
    NUM_KP = 14
    HM_HEIGHT = 160
    HM_WIDTH = 160
    alpha = 0.3
    image_path = '../demo_io/sample/1a98fcb21be0c72e0919fa98b1e3c8edd08cb70eT.jpg'
    hm_paths = ['../demo_io/sample/' + str(elem) + '.jpg' for elem in range(NUM_KP)]



    org_img = cv2.imread(image_path)
    for file_name in hm_paths:
        org_hm = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        img_with_hm = fuse_image_with_hm(org_img, org_hm, alpha)


        cv2.imshow('fuse_image_with_hm', img_with_hm)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
