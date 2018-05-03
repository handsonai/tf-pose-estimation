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
import time
import requests

mapping = {
    "Nose": 0,
    "Neck": 1,
    "Right_Shoulder": 2,
    "Right_Elbow": 3,
    "Right_Wrist": 4,
    "Left_Shoulder": 5,
    "Left_Elbow": 6,
    "Left_Wrist": 7,
    "Right_Hip": 8,
    "Right_Knee": 9,
    "Right_Ankle": 10,
    "Left_Hip": 11,
    "Left_Knee": 12,
    "Left_Ankle": 13,
    "Right_Eye": 14,
    "Left_Eye": 15,
    "Right_Ear": 16,
    "Left_Ear": 17  
}

class Part:
    def __init__(self, x, y, score):
        self.x = x
        self.y = y
        self.score = score

# parameters
test_image_path = './images_compare/image0178.jpg'
model = 'mobilenet_thin'

# script
#w, h = model_wh('432x368')
#e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
#image = common.read_imgfile(test_image_path, None, None)
#humans = e.inference(image, scales=[None])




start_time = time.time()
for i in range(0,20):
    time.sleep(1)
    response = requests.get("http://192.168.0.140:3333/data")
    json_data = response.json()
    humans = json_data["data"]["results"]["humans"]

    for human in humans:

        # 18 (0,0) parts:
        parts = [Part(0, 0, 0) for i in range(18)]

        for part in human:
            name = part[0]
            index = mapping[name]
            parts[index] = Part(part[1], part[2], 0)

        match_image_path = compare_pose(parts, verbose=True, use_second_match=True)

        if match_image_path is not None:
            print(match_image_path)

            # plot source and dest image:
            f, ax = plt.subplots(1)
            test_img = mpimg.imread(match_image_path)
            ax.imshow(test_img)
            plt.show()
