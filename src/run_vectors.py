import argparse
import logging
import time
import glob
import ast
import os
#import dill
import json

import common
import cv2
import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
from common import CocoPart

from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run by folder')
    parser.add_argument('--folder', type=str, default='./images/')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    args = parser.parse_args()
    scales = ast.literal_eval(args.scales)

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    files_grabbed = glob.glob(os.path.join(args.folder, '*.jpg'))
    all_humans = dict()
    for i, file in enumerate(files_grabbed):
        # estimate human poses from a single image !
        image = common.read_imgfile(file, None, None)
        t = time.time()
        humans = e.inference(image, scales=scales)
        elapsed = time.time() - t

        # translate output to a vector
        num_bodyparts = len(CocoPart.__members__)
        vector = np.zeros(num_bodyparts * 2)
        for human in humans:
            for body_ix in range(0, num_bodyparts):
                if body_ix in human.body_parts:
                    vector[body_ix] = human.body_parts[body_ix].x
                    vector[body_ix + 1] = human.body_parts[body_ix].y

        # save vector to a json file
        file_name, file_ext = os.path.splitext(file)
        file_name = os.path.basename(file_name)
        destination = os.path.join(args.folder, 'vectors')
        if not os.path.exists(destination):
            os.makedirs(destination)
        with open(os.path.join(destination, file_name + '.json'), 'w') as json_file:
            json.dump(list(vector), json_file)
