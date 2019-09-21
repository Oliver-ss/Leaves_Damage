'''*************************************************************************
	> File Name: visualize.py
	> Author: yuansong
	> Mail: yuansongwx@outlook.com
	> Created Time: Wed 18 Sep 2019 12:46:13 PM EDT
 ************************************************************************'''
import cv2
import json
import numpy as np
import os
from tqdm import tqdm
COLORS = ([255,0,0], [0,255,0], [0,0,255])
FONT = cv2.FONT_HERSHEY_SIMPLEX
INDEX = {
        'margin': 0, 'interior': 1, 'skel': 2, 'scrap':3, 'stipp': 4,
        'blotch': 5, 'serp': 6, 'undef': 7, 'normmar': 8, 'normint': 9,
        }



def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=10):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            img = cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                img = cv2.line(img,s,e,color,thickness)
            i+=1
    return img

def drawpoly(img,pts,color,thickness=1,style='dotted',):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        img = drawline(img,s,e,color,thickness,style)
    return img

def drawrect(img,pt1,pt2,color,thickness=1,style='dotted'):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])] 
    img = drawpoly(img,pts,color,thickness,style)
    return img

def save_img(pred, gt, save_folder='train_log/test_img/', thres=(0.4, 0.6, 0.5)):
    img_names = list(gt.keys())
    for i in tqdm(range(len(img_names))):
        img_file = img_names[i]
        img = cv2.imread(img_file)
        img = img[:,:,::-1]
        for box in pred[img_file]:
            xs, ys, xe, ye = box[:4]
            confidence = box[4]
            label = int(box[5])
            if confidence > thres[label]:
                img = cv2.rectangle(img, (int(xs), int(ys)), (int(xe), int(ye)), COLORS[label], 2)
                img = cv2.putText(img, str(np.round(confidence*10,2)), (int(xs), int(ys)),
                        FONT, 1, COLORS[label], 1, cv2.LINE_AA)

        for box in gt[img_file]:
            xs, ys, w, h = box['rect']
            xe, ye = xs + w, ys + h
            label = INDEX[box['label']]
            if label > 6:
                continue
            else:
                label = min(label, 2)
            img = drawrect(img, (int(xs), int(ys)), (int(xe), int(ye)), COLORS[label], 2, style='dash')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        name = img_file.split('/')[-1]
        cv2.imwrite(save_folder + name, img)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("thres1", type=float)
    parser.add_argument("thres2", type=float)
    parser.add_argument("thres3", type=float)
    parser.add_argument("pred", type=str)
    args = parser.parse_args()
    thres = (args.thres1, args.thres2, args.thres3)

    with open(args.pred) as f:
        pred = json.load(f)

    with open('../../Data/Labels/test_full.json') as f:
        gt = json.load(f)
    
    save_folder = 'train_log/test_img/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_img(pred, gt, save_folder, thres)

