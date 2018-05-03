# eigenpose.py
# 
# performs PCA on processed pose data.
# needs a json file with 'openpose-style' annotated and normalized poses,
# run read_annotations.py to generate this from coco annotation data.

# imports
import numpy as np
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# parameters
data_file = './processed/normalized_train2017.json'

show_lines = [
    [1, 16],
    [16, 17],
    [17, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [1, 5],
    [5, 6],
    [6, 7],
    [2, 8],
    [5, 11],
    [8, 11],
    [8, 9],
    [9, 10],
    [11, 12],
    [12, 13]
]

# script
# load data
with open(data_file) as infile:
    data = json.load(infile)

# transform to numpy array
matrix = []
for body in data:
    row  = []
    for part in body:
        row.append(part['x'])
        row.append(part['y'])
    matrix.append(row)

np_matrix = np.array(matrix)

# apply PCA:
pca = PCA(n_components=36)
pca.fit(np_matrix)

# retreive eigenvectors:

eigen_matrix = pca.inverse_transform(np.eye(36))
eigen_matrix_negative = pca.inverse_transform(-np.eye(36))

# plot:
num_plots_y = 3
num_plots_x = 6
f, axarr = plt.subplots(num_plots_y, num_plots_x * 2)
for iy in range(num_plots_y):
    for ix in range(num_plots_x):
        # plot +
        ax = axarr[iy, ix*2]
        ax.set_xlim([-1,1.5])
        ax.set_ylim([-1,1.5])
        ax.axis('off')
        ax.invert_yaxis()
        i = num_plots_x * iy + ix
        ax.set_title('comp {} +'.format(i))
        vector = eigen_matrix[i, :]
        for line in show_lines:
            x = [vector[line[0]*2], vector[line[1]*2]]
            y = [vector[line[0]*2+1], vector[line[1]*2+1]]
            ax.plot(x, y)

        # plot -
        # I know, not very DRY
        ax = axarr[iy, ix*2+1]
        ax.set_xlim([-1,1.5])
        ax.set_ylim([-1,1.5])
        ax.axis('off')
        ax.invert_yaxis()
        i = num_plots_x * iy + ix
        ax.set_title('comp {} -'.format(i))
        vector = eigen_matrix_negative[i, :]
        for line in show_lines:
            x = [vector[line[0]*2], vector[line[1]*2]]
            y = [vector[line[0]*2+1], vector[line[1]*2+1]]
            ax.plot(x, y)

plt.show()
