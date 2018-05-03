# extract_body_parts_store.py
#
# Reads a folder (inc subfolders) with images, finds and analyzes poses in the
# images, stores results.

# parameters
destination_path = './processed/data.json'
source_path = './images_compare'

# imports
import argparse
import time
import ast
import os
import sys
import common
import cv2
import numpy as np
import csv
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
from normalize_parts import normalize_parts
import json

# functions
def float_or_nan(x):
    try:
        return float(x)
    except ValueError:
        return float('nan')

# script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    args = parser.parse_args()

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # collect files
    if not os.path.isfile(source_path) and not os.path.isdir(source_path):
        print('No valid file or directory path given.')
        sys.exit()

    source_images = []
    if os.path.isfile(source_path):
        source_images.append({
            'path': source_path,
            'dirname': os.path.basename(source_path)
        })
    else:
        for root, dirs, files in os.walk(source_path):
            for filename in files:
                source_images.append({
                    'path': os.path.join(root, filename),
                    'dirname': os.path.basename(root)
                })

    # start analyzing
    data = []
    t = time.time()
    for source_image in source_images:
        print('Analyzing ' + source_image['path'])

        # estimate human poses from a single image
        image = common.read_imgfile(source_image['path'], None, None)
        try:
            humans = e.inference(image, scales=[None])
        except Exception:
            print("Couldn't load or analyze image. Skipping...")
            continue

        for human in humans:
            print('Found pose.')
            # normalize coordinates
            parts_norm = normalize_parts(human.body_parts, all_points_needed = True)

            if parts_norm is None:
                print('Pose missing points, skipping.')
            else:
                source_image['parts'] = parts_norm
                data.append(source_image)

    elapsed = time.time() - t
    print('inference done: analyzed {} images in {} seconds.'.format(len(source_images), round(time.time()-t)))
    print('Found {} full bodies. Saving...'.format(len(data)))

    with open(destination_path, 'w') as outfile:
        json.dump(data, outfile)
