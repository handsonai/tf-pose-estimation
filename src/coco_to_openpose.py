# coco_to_openpose.py
#
# coco_to_openpose(): Function for transforming a list of 51 values denoting
# x,y,score triplets as found in the COCO keypoints dataset annotations to a
# list of parts similar to what is returned by he TfPoseEstimator.inference
# Function.

mapping = {
    0: 0,
    1: 15,
    2: 14,
    3: 17,
    4: 16,
    5: 5,
    6: 2,
    7: 6,
    8: 3,
    9: 7,
    10: 4,
    11: 11,
    12: 8,
    13: 12,
    14: 9,
    15: 13,
    16: 10
}

class Part:
    def __init__(self, x, y, score):
        self.x = x
        self.y = y
        self.score = score

def coco_to_openpose(values):
    parts_dict = {}
    for i in range(0, 51, 3):
        x = values[i]
        y = values[i+1]
        score = 1
        coco_part_number = i/3
        op_part_number = mapping[coco_part_number]
        parts_dict[op_part_number] = Part(x, y, score)

    # generate part 1, between (openpose) parts 2 and 5:
    x1 = 0.5 * (parts_dict[2].x + parts_dict[5].x)
    y1 = 0.5 * (parts_dict[2].y + parts_dict[5].y)
    score1 = 0.5 * (parts_dict[2].score + parts_dict[5].score)
    parts_dict[1] = Part(x1, y1, score1)

    # turn dict into list:
    parts = []
    for i in range(18):
        parts.append(parts_dict[i])

    return parts
