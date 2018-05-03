# read_annotations.py
#
# Reads COCO annotations, converts them to 'openpose-style', normalizes and
# saves them.

# imports
import json
from coco_to_openpose import coco_to_openpose
from normalize_parts import normalize_parts

# parameters
annotations_file_path = './annotations_trainval2017/annotations/person_keypoints_train2017.json'
data_save_file = './processed/normalized_train2017.json'

# script
with open(annotations_file_path) as in_file:
    data = json.load(in_file)

images = data['annotations']
print('found {} images'.format(len(images)))

data = []
for image in images:

    if image['num_keypoints'] == 17:
        parts = coco_to_openpose(image['keypoints'])
        parts_norm = normalize_parts(parts, all_points_needed = True)
        data.append(parts_norm)

print('Found {} full bodies. Saving...'.format(len(data)))

with open(data_save_file, 'w') as outfile:
    json.dump(data, outfile)
