'''*************************************************************************
	> File Name: show_pic.py
	> Author: yuansong
	> Mail: yuansongwx@outlook.com
	> Created Time: Mon 16 Sep 2019 11:49:49 AM EDT
 ************************************************************************'''
import cv2
import json

FONT = cv2.FONT_HERSHEY_SIMPLEX

with open('../../Data/Labels/label_test_new.json') as f:
    data = json.load(f)

for k in data.keys():
    img = cv2.imread(k)
    hh, ww = img.shape[:2]
    img = cv2.resize(img, (int(ww/5), int(hh/5)))
    for box in data[k]:
        if box['label'] == 'normmar' or box['label'] == 'normint':
            continue
        x, y, w, h = box['rect']
        img = cv2.rectangle(img, (int(x/5),int(y/5)), (int(x/5+w/5), int(y/5+h/5)), (0,0,0), 2)
        img = cv2.putText(img, box['label'], (int(x/5), int(y/5)),
                    FONT, 1, (0, 0, 0), 1, cv2.LINE_AA)
    #img = cv2.resize(img, (int(ww/5), int(hh/5)))
    cv2.imshow('img', img)
    c = chr(cv2.waitKey(0) & 0xff)
    if c == 'q':
        exit()
