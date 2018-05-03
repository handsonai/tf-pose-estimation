# test_compare_pose.py
#
#

# imports
import cv2
import numpy as np
import csv
import common
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
from normalize_parts import normalize_parts
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from compare_pose import compare_pose
import sys

# parameters
test_image_path = './images_compare/image0178.jpg'
model = 'mobilenet_thin'

# script
w, h = model_wh('432x368')
e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
image = common.read_imgfile(test_image_path, None, None)
humans = e.inference(image, scales=[None])

for human in humans:
    print('Found pose.')
    match_image_path = compare_pose(human.body_parts, verbose=True, use_second_match=True)

    if match_image_path is not None:
        # plot source and dest image:
        f, (ax1, ax2) = plt.subplots(1, 2)

        test_img = mpimg.imread(test_image_path)
        ax1.imshow(test_img)
        ax1.set_title('Test image')

        match_img = mpimg.imread(match_image_path)
        ax2.imshow(match_img)
        ax2.set_title('Best match')

        plt.show()
