""" 
Ensembling methods for object detection.
    # USAGE: edit a, b, c as the csv file you want to ensemble
    # If you want to change size of width, height, modify convert_ratio. 
    # There should be './stage_2_sample_submission.csv' in your working directory.
    # Output will be './ensemble_answer.csv'
"""

import csv
import pandas as pd

convert_ratio = 0.93
convert_ratio = 1

""" 
General Ensemble - find overlapping boxes of the same class and average their positions
while adding their confidences. Can weigh different detectors with different weights.
No real learning here, although the weights and iou_thresh can be optimized.
Input: 
 - dets : List of detections. Each detection is all the output from one detector, and
          should be a list of boxes, where each box should be on the format 
          [box_x, box_y, box_w, box_h, class, confidence] where box_x and box_y 
          are the center coordinates, box_w and box_h are width and height resp.
          The values should be floats, except the class which should be an integer.
 - iou_thresh: Threshold in terms of IOU where two boxes are considered the same, 
               if they also belong to the same class.
               
 - weights: A list of weights, describing how much more some detectors should
            be trusted compared to others. The list should be as long as the
            number of detections. If this is set to None, then all detectors
            will be considered equally reliable. The sum of weights does not
            necessarily have to be 1.
Output:
    A list of boxes, on the same format as the input. Confidences are in range 0-1.
"""


def GeneralEnsemble(dets, iou_thresh=0.5, weights=None):
    assert(type(iou_thresh) == float)

    ndets = len(dets)
    # print(f'ndets: {ndets}')

    # Judge whether empty predict
    empty_pred_cnt = 0
    pop_list = []
    for i in range(ndets):
        if dets[i] == [[]]:
            pop_list.append(i)
            # dets.pop(i)
            # weights.pop(i)
            ndets -= 1
            empty_pred_cnt += 1

    for i in reversed(pop_list):
        dets.pop(i)
        if weights is not None:
            weights.pop(i)

    if ndets == 0:
        return []

    if weights is None:
        w = 1/float(ndets + empty_pred_cnt)
        weights = [w]*(ndets + empty_pred_cnt)
    else:
        assert(len(weights) == ndets)

        s = sum(weights)
        for i in range(0, len(weights)):
            weights[i] /= s

    out = list()
    used = list()

    for idet in range(0, ndets):
        det = dets[idet]
        for box in det:
            if box in used:
                continue

            used.append(box)
            # Search the other detectors for overlapping box of same class
            found = []
            for iodet in range(0, ndets):
                odet = dets[iodet]

                if odet == det:
                    continue

                bestbox = None
                bestiou = iou_thresh
                for obox in odet:
                    if not obox in used:
                        # Not already used
                        if box[4] == obox[4]:
                            # Same class
                            iou = computeIOU(box, obox)
                            if iou > bestiou:
                                bestiou = iou
                                bestbox = obox

                if not bestbox is None:
                    w = weights[iodet]
                    found.append((bestbox, w))
                    used.append(bestbox)

            # Now we've gone through all other detectors
            if len(found) == 0:
                new_box = list(box)
                new_box[5] /= ndets

                # ME reorganized
                conf = new_box[5]
                xc = new_box[0]
                yc = new_box[1]
                bw = new_box[2]
                bh = new_box[3]
                x_left = xc - bw/2
                y_top = yc - bh/2
                new_box = [conf, x_left, y_top, bw, bh]
                out.append(new_box)
            else:
                allboxes = [(box, weights[idet])]
                allboxes.extend(found)

                xc = 0.0
                yc = 0.0
                bw = 0.0
                bh = 0.0
                conf = 0.0

                wsum = 0.0
                for bb in allboxes:
                    w = bb[1]
                    wsum += w

                    b = bb[0]
                    xc += w*b[0]
                    yc += w*b[1]
                    bw += w*b[2]
                    bh += w*b[3]
                    conf += w*b[5]

                xc /= wsum
                yc /= wsum
                bw /= wsum
                bh /= wsum

                # new_box = [xc, yc, bw, bh, box[4], conf]

                x_left = xc - bw/2
                y_top = yc - bh/2
                new_box = [conf, x_left, y_top, bw, bh]

                out.append(new_box)
    return out


def getCoords(box):
    x1 = float(box[0]) - float(box[2])/2
    x2 = float(box[0]) + float(box[2])/2
    y1 = float(box[1]) - float(box[3])/2
    y2 = float(box[1]) + float(box[3])/2
    return x1, x2, y1, y2


def computeIOU(box1, box2):
    x11, x12, y11, y12 = getCoords(box1)
    x21, x22, y21, y22 = getCoords(box2)

    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersect_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)

    iou = intersect_area / (box1_area + box2_area - intersect_area)
    return iou

# return: [[box_x, box_y, box_w, box_h, class, confidence] * n1]
# bbox_str: [(conf, x1, y1, width, height) * n2]


def str2list(bbox_str, convert_ratio=1):
    if bbox_str == '':  # empty str
        return [[]]
    res = []
    cnt = 0
    bbox_list = bbox_str.split(' ')

    while(cnt < len(bbox_list)):
        confidence = float(bbox_list[0 + cnt])
        x = float(bbox_list[1 + cnt])
        y = float(bbox_list[2 + cnt])
        w = float(bbox_list[3 + cnt])
        h = float(bbox_list[4 + cnt])
        xc = x + w/2
        yc = y + h/2

        # Convert
        w = float(bbox_list[3 + cnt]) * convert_ratio
        h = float(bbox_list[4 + cnt]) * convert_ratio

        temp = [xc, yc, w, h, 0, confidence]
        res.append(temp)
        cnt += 5

    return res

    # for i in range(len(bbox_list)):
    #     if i % 5 == 0:


def csv2list(csv_path):
    with open(csv_path, 'r') as f:
        file = csv.reader(f)
        my_list = list(file)
    return my_list

# Get the bboxes from one same id


def combine_by_id(list_of_csv, id,  weights=None):
    total_csv = len(list_of_csv)
    csv_bboxes = []
    for i in range(len(list_of_csv)):
        for csv in list_of_csv[i]:
            if id == csv[0]:
                id_bboxes = str2list(csv[1], convert_ratio=convert_ratio)
                # print(id_bboxes)
                csv_bboxes.append(id_bboxes)

    # print(csv_bboxes)
    ens = GeneralEnsemble(csv_bboxes, weights=weights)
    print(ens)
    return ens


if __name__ == "__main__":
    # Toy example
    # dets = [
    #         [[0.1, 0.1, 1.0, 1.0, 0, 0.9], [1.2, 1.4, 0.5, 1.5, 0, 0.9]],
    #         [[0.2, 0.1, 0.9, 1.1, 0, 0.8]],
    #         [[5.0,5.0,1.0,1.0,0,0.5]],
    #         [[]]
    #        ]
    # dets = [
    #         [[]],
    #         [[]],
    #         [[]],
    #         [[]]
    #        ]
    # ens = GeneralEnsemble(dets, weights = [1.0, 0.1, 0.5, 0])
    # print(ens)

    # USAGE: edit a, b, c as the csv file you want to ensemble
    a = './1_mask-rcnn-swinT_lr0.001batch2threshold0.7epoch10_0.14308.csv'
    b = './swinT_ep15_ALLDATA_batchsize2_conf0.7_0.13571.csv'
    # c = './swinT_ep15_val target1_batchsize1_conf0.7_0.13376.csv'
    # d = './detectors_cascade_rcnn_random-cropconf0.5_0.06992.csv'
    # e = './2_mask-rcnn-swinT_lr0.001batch2threshold0.7epoch100.91_0.11715.csv'
    # f = './swinT_ep10_val target1_batchsize1_conf0.7_0.12141.csv'

    a = csv2list(a)
    b = csv2list(b)
    # c = csv2list(c)
    # d = csv2list(d)
    # e = csv2list(e)
    # f = csv2list(f)
    list_of_csv = [a, b]
    # id = '00271e8e-aea8-4f0a-8a34-3025831f1079'
    # id_ans = combine_by_id(list_of_csv, id)

    df = pd.read_csv('stage_2_sample_submission.csv')
    for id in df['patientId']:
        str_id_ans = ''
        temp_list = []
        id_ans = combine_by_id(list_of_csv, id)

        for ans in id_ans:
            for s in ans:
                temp_list.append(s)
        temp_str = ' '.join(str(i) for i in temp_list)
        df['PredictionString'].loc[df['patientId'] == id] = temp_str

    df.to_csv('./ensemble_answer.csv', encoding='utf-8', index=False)
