'''*************************************************************************
	> File Name: train_test_split.py
	> Author: yuansong
	> Mail: yuansongwx@outlook.com
	> Created Time: Wed 11 Sep 2019 11:37:48 AM EDT
 ************************************************************************'''
from trans_label import get_partition
import json

if __name__ == "__main__":
    with open('../../Data/Labels/label_no_overlap.json') as f:
        data = json.load(f)
    train, vt = get_partition(data, 0.3)
    test, val = get_partition(vt, 0.3)
    with open('../../Data/Labels/label_train_new.json', 'w') as f:
        json.dump(train, f)
    with open('../../Data/Labels/label_val_new.json', 'w') as f:
        json.dump(val, f)
    with open('../../Data/Labels/label_test_new.json', 'w') as f:
        json.dump(test, f)




