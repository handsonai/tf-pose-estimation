# compare_pose.py
#
# Function for comparing a pose with a large number of pre-analyzed poses.
# Assumes you ran the script extract_body_parts_image.py, so that a data csv
# file is present at the path './processed/data.csv'
# (for this run extract_body_parts_store.py)

# parameters
poses_data_path = './processed/data.json'

# imports
import numpy as np
import json
from normalize_parts import normalize_parts
from scipy import spatial

# Do some initial work upon import:
# load data
with open(poses_data_path) as infile:
    data = json.load(infile)

# organize
coord_data = []
image_paths = []
for image in data:
    image_paths.append(image['path'])
    coord_row = []
    for part in image['parts']:
        coord_row.append(part['x'])
        coord_row.append(part['y'])
    coord_data.append(coord_row)

# make KDTree
tree = spatial.KDTree(coord_data)

# function:
def compare_pose(parts, verbose=False, use_second_match=False): # use second neighbor to test with the same images
    # normalize:
    parts_norm = normalize_parts(parts, all_points_needed = True)

    if parts_norm is None:
        if verbose:
            print('Pose is missing parts, can\'t compare.')
        return None

    if verbose:
        print('Full pose. Comparing...')

    vector = []
    for part in parts_norm:
        vector.append(part['x'])
        vector.append(part['y'])

    if use_second_match:
        num_neighbors = 2
    else:
        num_neighbors = 1

    result = tree.query(vector, k=num_neighbors)

    if use_second_match:
        found_index = result[1][1]
    else:
        found_index = result[1]

    file_path = data[found_index]['path']
    if verbose:
        print('Closest pose was found in image: {}'.format(file_path))
    return file_path
