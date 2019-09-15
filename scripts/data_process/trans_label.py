'''*************************************************************************
	> File Name: dataset.py
	> Author: Your Name (edit this in the init.vim file)
	> Mail: Your Mail@megvii.com
	> Created Time: Tue Sep  3 14:20:52 2019
 ************************************************************************'''
import json
import os
import numpy as np
import warnings
import glob

warnings.filterwarnings('ignore')

ROOT = '../../Data/'
IMAGE_ROOT = os.path.join(ROOT, 'Images')
ANNOTATION_ROOT = os.path.join(ROOT, 'Annotations')
LABEL_ROOT = os.path.join(ROOT, 'Labels')
json_files = glob.glob(r'../../Data/Annotations/Damage/*.json')

if not os.path.exists(LABEL_ROOT):
    os.makedirs(LABEL_ROOT)

def readDataFromFile(jasonFilename, imageDir, species='', class_type='DamageType'):
    data = []
    with open(jasonFilename, 'r') as file:
        jason = json.load(file)
        imgAnnoList = jason['_via_img_metadata'].values()
        for imgAnno in imgAnnoList:
            imgBasename = imgAnno['filename']
            imgFilename = os.path.join(imageDir, species, imgBasename)
            regionList = imgAnno['regions']
            for region in regionList:
                rectangle = region['shape_attributes']
                if rectangle['name'] == 'rect':
                    xs, ys = rectangle['x'], rectangle['y']
                    w, h = rectangle['width'], rectangle['height']
                    xe, ye = xs + w, ys + h
                    label = region['region_attributes'][class_type]
                    record = {'imageFilename': imgFilename,
                              'species': species, 'label': label,
                              'rect': (xs, ys, w, h)}
                    data.append(record)
    return data

def get_all_data(json_files, imageDir):
    data = []
    for j in json_files:
        species = j.split('/')[-1][:-6]
        data.extend(readDataFromFile(j, imageDir, species=species))
    return data

def trans_label(data):
    label = {}
    for d in data:
        if d['imageFilename'] not in label.keys():
            label[d['imageFilename']] = []
        label[d['imageFilename']].append(d)
    return label

def filter_classes(label):
    label_new = {}
    for k in label.keys():
        num = 0
        for box in label[k]:
            l = box['label']
            if l == 'undef' or l == 'normmar' or l == 'normint':
                num += 1
        if num < len(label[k]):
            label_new[k] = label[k]
    return label_new

def get_partition(label, ratio=0.2):
    keys = list(label.keys())
    #np.random.seed(42)
    np.random.shuffle(keys)
    train, val = {}, {}
    for i, key in enumerate(keys):
        if i < ratio * len(keys):
            val[key] = label[key]
        else:
            train[key] = label[key]
    return train, val

def iou(box_a, box_b):
    box_a = [box_a[0], box_a[1], box_a[0] + box_a[2], box_a[1] + box_a[3]]
    box_b = [box_b[0], box_b[1], box_b[0] + box_b[2], box_b[1] + box_b[3]]
    max_xy = np.minimum(box_a[2:], box_b[2:])
    min_xy = np.maximum(box_a[:2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    inter_area = inter[0] * inter[1]
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_a[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area
    return inter_area / union_area

def filter_overlap_boxes(label):
    label_new = {}
    for k in label.keys():
        n = len(label[k])
        if n  == 1:
            label_new[k] = label[k]
        else:
            overlap = []
            ans = []
            for i in range(n):
                if i in overlap:
                    continue
                else:
                    for j in range(n):
                        if iou(label[k][i]['rect'], label[k][j]['rect']) > 0.35 and label[k][i]['label'] == label[k][j]['label']:
                            overlap.append(j)
                    ans.append(label[k][i])
            label_new[k] = ans
    return label_new

if __name__ == "__main__":
    tr_path = os.path.join(LABEL_ROOT, 'label_train_no_overlap.json')
    val_path = os.path.join(LABEL_ROOT, 'label_val_no_overlap.json')
    path = os.path.join(LABEL_ROOT, 'label_no_overlap.json')
    
    data = get_all_data(json_files, IMAGE_ROOT)
    label= trans_label(data)
    label = filter_classes(label)
    label = filter_overlap_boxes(label)
    with open(path, 'w') as f:
        json.dump(label, f)

    tr, va = get_partition(label)
    with open(tr_path, 'w') as f:
        json.dump(tr, f)
    with open(val_path, 'w') as f:
        json.dump(va, f)
