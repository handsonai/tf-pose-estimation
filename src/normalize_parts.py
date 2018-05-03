# normalize_parts.py
#
# Function for transforming the body part coordinates in such a way that
# the coordinates become position- and size-independent, to facilitate comparing
# poses across different images.
#
# The origin of the new coordinate system is between points 1, 8 and 11, this is
# close to the center of gravity of a person standing upright.
#
# The coordinates are scaled so that the average distance of lines 1-8 and 1-11
# is 1.0. The lenght of these lines is somewhat invariant to the orientation of
# the subject, except when bending towards or away from the camera.
#
# Input: body_parts object as returned by the TfPoseEstimator.inference Function
#
# Output: list of dictionaries containing x, y and score.

import math

def normalize_parts(parts, all_points_needed = True):
    # validation:
    if all_points_needed:
        # we need all points, otherwise discard
        if len([p for p in parts]) != 18:
            return None
    else:
        # we need at least points 1, 8 and 11 for normalization
        if 1 not in parts or 8 not in parts or 11 not in parts:
            return None

    # 1. get origin for translation
    origin_x = (parts[1].x + parts[8].x + parts[11].x) / 3
    origin_y = (parts[1].y + parts[8].y + parts[11].y) / 3

    # 2. get scale
    line1 = math.sqrt(pow(parts[8].x - parts[1].x, 2) + pow(parts[8].y - parts[1].y, 2))
    line2 = math.sqrt(pow(parts[11].x - parts[1].x, 2) + pow(parts[11].y - parts[1].y, 2))
    scale = 0.5 * (line1 + line2)

    # 3. apply transformation
    parts_norm = []

    for i in range(18):
        try:
            part_norm = {
                'x': (parts[i].x - origin_x) / scale,
                'y': (parts[i].y - origin_y) / scale,
                'score': parts[i].score
            }
        except KeyError:
            part_norm = {'x': None, 'y': None, 'score': None}
        parts_norm.append(part_norm)

    return parts_norm
