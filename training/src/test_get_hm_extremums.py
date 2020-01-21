import numpy as np
from scipy.ndimage.filters import gaussian_filter


def get_hm_extremums(heatmap, gauss_sigma):
    # filt
    heatmap = gaussian_filter(heatmap, sigma=gauss_sigma)
    # padding
    heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode='constant')
    heatmap_center = heatmap_with_borders[1:heatmap_with_borders.shape[0] - 1, 1:heatmap_with_borders.shape[1] - 1]
    heatmap_left = heatmap_with_borders[1:heatmap_with_borders.shape[0] - 1, 2:heatmap_with_borders.shape[1]]
    heatmap_right = heatmap_with_borders[1:heatmap_with_borders.shape[0] - 1, 0:heatmap_with_borders.shape[1] - 2]
    heatmap_up = heatmap_with_borders[2:heatmap_with_borders.shape[0], 1:heatmap_with_borders.shape[1] - 1]
    heatmap_down = heatmap_with_borders[0:heatmap_with_borders.shape[0] - 2, 1:heatmap_with_borders.shape[1] - 1]
    # peak mask with padding
    heatmap_peaks = (heatmap_center > heatmap_left) & \
                    (heatmap_center > heatmap_right) & \
                    (heatmap_center > heatmap_up) & \
                    (heatmap_center > heatmap_down)
    # peak mask without padding
    heatmap_peaks = heatmap_peaks[1:heatmap_center.shape[0] - 1, 1:heatmap_center.shape[1] - 1]
    return heatmap_peaks

hm = np.random.random([10,10])
print(hm)
hm_ext_mask = get_hm_extremums(hm, 1)
print(hm_ext_mask)
hm_ext_vals = hm[hm_ext_mask]
hm_ext_vals.sort()
print(hm_ext_vals)