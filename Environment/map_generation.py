"""

Different resolution Ypacaraí Map Generator

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import cv2
from skimage.transform import rescale, resize

def generate_gaussian_maps(map, peaks, sigma):

    importance_map = np.zeros(map.shape, dtype=float)

    for i in range(peaks.shape[0]):

        var = multivariate_normal(mean=[peaks[i, 0], peaks[i, 1]], cov=[[sigma, 0], [0, sigma]])
        x = np.linspace(0, map.shape[0], map.shape[0])
        y = np.linspace(0, map.shape[1], map.shape[1])
        X, Y = np.meshgrid(x, y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        g = var.pdf(pos)
        importance_map += g.T

    importance_map = importance_map / np.max(importance_map)
    importance_map = np.clip(importance_map, 0.5, 1) * map

    return importance_map


resolution = [1, 2, 3, 4]
peaks = np.array([[3, 3], [8, 3], [11, 7]])
sigma = 2.5

min_map = np.genfromtxt('YpacaraiMap_big.csv', delimiter=',')
maps = []
importance_maps = []

init_points = np.array([[5,6],[11,12],[17,19],[23,25]])

for r in resolution:

    resized = rescale(min_map, 0.06*r, anti_aliasing=True, order = 3)
    resized[resized < 0.4*255] = 0
    resized[resized >= 0.4*255] = 1
    #resized[init_points[r-1,0], init_points[r-1,1]] = np.nan
    importance_map = generate_gaussian_maps(resized, peaks*r, sigma*r*r)

    np.savetxt('map_{}.csv'.format(r), resized, delimiter=',')
    np.savetxt('importance_map_{}.csv'.format(r), importance_map, delimiter=',')

    maps.append(np.copy(resized))
    importance_maps.append(np.copy(importance_map))

fig, axs = plt.subplots(2, 4)

for i in range(len(resolution)):

    axs[0][i].imshow(maps[i])
    axs[1][i].imshow(importance_maps[i])

plt.show()




