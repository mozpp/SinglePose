import cv2
import numpy as np

def fuse_image_with_hm(image_path, hm_paths, hm_h, hm_w, alpha, plot=False):
    '''
    show img with hms as a mask
    '''
    org_img = cv2.imread(image_path)
    h, w = org_img.shape[:2]

    hms = np.zeros([hm_h, hm_w, len(hm_paths)])
    for hm_ind, file_name in enumerate(hm_paths):
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        hms[:, :, hm_ind] = img

    fused_hm = np.mean(hms, axis=-1)
    fused_hm = cv2.resize(fused_hm, (w, h))
    fused_hm = fused_hm / np.max(fused_hm) * 255
    fused_hm = fused_hm.astype(np.uint8)
    fused_hm = np.stack(3 * [fused_hm], axis=-1)

    img_with_hm = alpha * org_img + (1 - alpha) * fused_hm

    if plot:
        cv2.imshow('fused_hm', fused_hm)
        cv2.waitKey(0)

        cv2.imshow('fuse_image_with_hm', img_with_hm.astype(np.uint8))
        cv2.waitKey(0)

        cv2.destroyAllWindows()

if __name__ == '__main__':
    NUM_KP = 14
    HM_HEIGHT = 160
    HM_WIDTH = 160
    alpha = 0.3
    image_path = '../demo_io/sample/1a98fcb21be0c72e0919fa98b1e3c8edd08cb70eT.jpg'
    hm_paths = ['../demo_io/sample/' + str(elem) + '.jpg' for elem in range(NUM_KP)]

    fuse_image_with_hm(image_path, hm_paths, 160, 160, alpha, plot=True)


