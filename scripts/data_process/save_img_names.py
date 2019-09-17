'''*************************************************************************
	> File Name: save_img_names.py
	> Author: yuansong
	> Mail: yuansongwx@outlook.com
	> Created Time: Mon 16 Sep 2019 12:12:09 PM EDT
 ************************************************************************'''
import json

with open('../../Data/Labels/label_test_new.json') as f:
    data = json.load(f)

with open('../../Data/Labels/test_imgs.txt', 'w') as f:
    for key in data.keys():
        p = key.split('/')
        name = p[-2] + '/' + p[-1]
        f.write(name+'\n')
