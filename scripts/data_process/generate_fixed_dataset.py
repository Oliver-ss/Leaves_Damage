'''*************************************************************************
	> File Name: dataset.py
	> Author: Your Name (edit this in the init.vim file)
	> Mail: Your Mail@megvii.com
	> Created Time: Tue Sep  3 15:54:29 2019
 ************************************************************************'''
#!/usr/bin/env python3
import os
import torch
import torch.utils.data as data
import cv2
import numpy as np
from torchvision import transforms
from config import *
import json

INDEX = {
        'margin': 0, 'interior': 1, 'skel': 2, 'stipp': 3,
        'blotch': 4, 'serp': 5, 'scrap': 6, 'normmar': 7, 'normint': 8, 'undef': 9,
        }

LABEL = ['margin', 'interior', 'skel', 'stipp', 'blotch', 'serp', 'scrap', 'normmar', 'normint', 'undef']
FONT = cv2.FONT_HERSHEY_SIMPLEX

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def crop_img(image, boxes, labels, name):
    hh, ww = image.shape[:2]
    num = 0
    data = {}
    for box, label in boxes, labels:
        folder_name = LABEL[label]
        root = '../../Data/Image/Image_patches/'+folder_name+'/'
        if not os.path.exists((root)):
            os.makedirs(root)
        img_name = root + name.split("/")[-1][:-4] + '_' + str(num) + '.jpg'
        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        w = h =500
        left = max(int(center[0]-250), 0)
        top = max(int(center[1]-250), 0)
        right = min(int(left + w), ww)
        bottom = min(int(top + h), hh)
        rect = np.array([left, top, right, bottom])
        current_image = image[rect[1]:rect[3], rect[0]:rect[2], :].copy()
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        m1 = (rect[0] + 50 < centers[:, 0]) * (rect[1] + 50 < centers[:, 1])
        m2 = (rect[2] - 50 > centers[:, 0]) * (rect[3] - 50 > centers[:, 1])
        mask = m1 * m2
        current_boxes = boxes[mask, :].copy()
        current_labels = labels[mask]
        current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                          rect[:2])
        current_boxes[:, :2] -= rect[:2]

        current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                          rect[2:])
        current_boxes[:, 2:] -= rect[:2]

        data['name'] = img_name
        data[]

class Generator:
    '''
    input is image, target is annotations for every image
    '''
    def __init__(self, name, label_root):
        self.name = name
        self.label_root = label_root
        self.labels = load_json(label_root)
        self.ids = list(self.labels.keys())

    def pull_item(self, index):
        print(self.ids[index])
        try:
            img = cv2.imread(self.ids[index])
        except img is None:
            print(self.ids[index]+' does not exist')

        img = img[:,:,::-1].copy() # translate the image from BGR to RGB
        height, width, channels = img.shape
        gt = self.labels[self.ids[index]]
        boxes = []
        labels = []
        for g in gt:
            label = INDEX[g['label']]
            if label > 7:
                continue # omit the normal margin, normal interior and undefined damage type
            rect = g['rect']
            # to macth the transform format
            rect = [float(rect[0]), float(rect[1]), float(rect[0]+rect[2]), float(rect[1]+rect[3])]
            boxes.append(rect)
            labels.append(label)

        boxes, labels = np.array(boxes), np.array(labels)
        img_patches, boxes_patches, labels_patches = crop_img(img, boxes, labels, self.ids[index])
        return img_patches, boxes_patches, labels_patches

    def draw_boxes(self, index):
        img_patches, boxes_patches, labels_patches  = self.pull_item(index)
        n = len(img_patches)
        imgs = []
        for i in range(n):
            img = img_patches[i]
            boxes = boxes_patches[i]
            labels = labels_patches[i]
            for box, label in boxes, labels:
                x, y, xr, yr = box
                label = LABEL[int(label)]
                img = cv2.rectangle(img, (int(x), int(y)), (int(xr), int(yr)), (0,0,255), 3)
                img = cv2.putText(img, label, (int(x), int(y)),
                        FONT, 1, (0, 0, 0), 1, cv2.LINE_AA)
            imgs.append(np.array(img))
        return imgs


def show_samples(ds, num):
    for i in range(num):
        imgs = ds.draw_boxes(i)
        for img in imgs:
            cv2.imshow('img', img[:,:,::-1])
            c = chr(cv2.waitKey(0) & 0xff)
            if c == 'q':
                exit()


def save_images(ds, num):
    n = len(ds.ids)
    file = {}
    for i in range(num):
        img, target, _, _ = ds.pull_item(i%n)
        name = '../../Data/Images/Validation-3class/'+str(i)+'.jpg'
        img = img[:,:,::-1]
        cv2.imwrite(name, img)
        target = target.tolist()
        file[name] = []
        for t in target:
            xs, ys, xe, ye = t[:4]
            label = LABEL[int(t[4])]
            file[name].append({'rect': [xs*300, ys*300, (xe-xs)*300,(ye-ys)*300], 'label': label})
        #print(file)
    with open('../../Data/Labels/Validation-3class.json', 'w') as f:
        json.dump(file, f)

if __name__ == '__main__':
    from aug_test import SSDAugmentation
    import torch.utils.data as data
    #ds = Damage_Dataset('train', '../../Data/Labels/label_val_new.json', transform=SSDAugmentation())
    ds = Damage_Dataset('train', '../../Data/Labels/Validation-3class.json')
    show_samples(ds, 100)
    #save_images(ds, 100)
