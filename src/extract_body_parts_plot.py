# extract_body_parts_plot.py
#
# Reads a folder (inc subfolders) with images, finds and analyzes poses in the
# images, plots results.

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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def float_or_nan(x):
    try:
        return float(x)
    except ValueError:
        return float('nan')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--source', type=str, default='./images')
    parser.add_argument('--destination', type=str, default='./processed/data.csv')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    args = parser.parse_args()

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    source_path = args.source
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
    print(source_images)

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
        # print body parts:
        for human in humans:
            print('Found pose.')
            # normalize coordinates:
            parts_norm = normalize_parts(human.body_parts, all_points_needed = True)

            if parts_norm is None:
                print('Pose missing points, skipping.')
            else:
                x_inds = []
                y_inds = []
                labels = []
                for label in human.body_parts:
                    img = mpimg.imread(source_image['path'])
                    part = human.body_parts[label]
                    x_inds.append(part.x * img.shape[1])
                    y_inds.append(part.y * img.shape[0])
                    labels.append(label)

                # plot:

                fig, ax = plt.subplots(1)
                ax.imshow(img)
                ax.scatter(x_inds, y_inds)

                for label, x, y in zip(labels, x_inds, y_inds):
                    plt.annotate(
                        label,
                        xy=(x, y), xytext=(0, 5),
                        textcoords='offset points', va='bottom', color='red')

                ax.set_title('keypoints')

                plt.show()
