'''*************************************************************************
	> File Name: get_mAP.py
	> Author: yuansong
	> Mail: yuansongwx@outlook.com
	> Created Time: Mon 16 Sep 2019 11:29:36 PM EDT
 ************************************************************************'''
import numpy as np
import os
import json
INDEX = {
        'margin': 0, 'interior': 1, 'skel': 2, 'scrap':3, 'stipp': 4,
        'blotch': 5, 'serp': 6, 'undef': 7, 'normmar': 8, 'normint': 9,
        }

LABEL = ['margin', 'interior', 'skel', 'scrap', 'stipp', 'blotch', 'serp', 'undef', 'normmar', 'normint']

def parse_rec(filename):
    with open(filename) as f:
        data = json.load(f)
    objects = {0: [], 1: [], 2:[]}
    for k in data.keys():
        for box in data[k]:
            label = INDEX[box['label']]
            if label > 6: # omit last 3 classes
                continue
            obj = {}
            obj['name'] = box['imageFilename']
            x, y, w, h = box['rect']
            obj['bbox'] = [x, y, x+w, y+h]
            if w < 10 or h < 10:
                obj['difficult'] = True
            else:
                obj['difficult'] = False
            obj['label'] = min(2, label) # conclue other classes as 1 class

            objects[obj['label']].append(obj)
    return objects

if __name__ == "__main__":
    print(parse_rec("../../Data/Labels/label_test_new.json"))


