# dataset loader for pytorch
#
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import time
import pdb


class VisDrone(Dataset):
    def __init__(self, rootPath, transform=None, target_transform=None):
        self.rootPath = rootPath
        self.transform = transform
        self.target_transform = target_transform
        self.name = 'VisDrone'
       # modelDict = { 'train': 'VisDrone2019-DET-train', 
       #               'val': 'VisDrone2019-DET-val', 
       #               'test': 'VisDrone2019-DET-test-challenge'}
       # self.basePath = os.path.join(rootPath, modelDict[self.model])
        self.basePath =  rootPath
        self.txtFiles = os.listdir(self.basePath + '/annotations/')

    def __len__(self):
        return len(self.txtFiles) 

    def __getitem__(self, idx):
        # load image
        baseName = self.txtFiles[idx].split('.')[0]
        imgName = baseName + '.jpg'
        imgPath = os.path.join(self.basePath, 'images', imgName)
        img = cv2.imread(imgPath)
        #print(img.shape)
        hw= img.shape #NOTE cv2 image shape (hight, width)
        h = hw[0]
        w = hw[1]


        # load box labels
        # labels: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
        txtPath = os.path.join(self.basePath, 'annotations', self.txtFiles[idx])
        with open(txtPath, 'r') as f:
            labels = f.readlines()    
            labels = [ tt.rstrip('\n').split(',') for tt in labels ] 
            tmp = np.empty(shape=(len(labels), len(labels[0]))).tolist()
            for i in range(len(labels)):
                for j in range(len(labels[i])):
                    tmp[i][j] = float(labels[i][j])
            labels = np.asarray(tmp)


        # to (xmin, ymin, xmax, ymax)
        labels[:, 2] = labels[:, 0] + labels[:, 2]
        labels[:, 3] = labels[:, 1] + labels[:, 3]
        bboxes = labels[:, :4]
        labels = labels[:, 5]
        

        #t0 = time.time()
        if self.transform is not None:
            img, bboxes = self.transform(img, bboxes) 
        #t1 = time.time()
        
        bboxes, labels = self.filter(img, bboxes, labels)

        if self.target_transform:
            bboxes, labels = self.target_transform(bboxes, labels)
        if self.transform is None:
            img = torch.from_numpy(img.astype('float32')).permute(2, 0, 1)
            bboxes = torch.from_numpy(bboxes)
            labels = torch.from_numpy(labels)

        #delta = t1 - t0
        #print('min{},sec{}'.format((delta//60%60), (delta%60)))

        return img, bboxes, labels

    def filter(self, img, boxes, labels):
        shape = img.shape
        if len(shape) == 2:
            h, w = shape
        else:   # !!
            if shape[0] > shape[2]:   # HWC
                h, w = img.shape[:2]
            else:                     # CHW
                h, w = img.shape[1:]

        boxes_ = []
        labels_ = []
        for box, label in zip(boxes, labels):
            if min(box[2] - box[0], box[3] - box[1]) <= 0:
                continue
            if np.max(boxes) < 1 and np.sqrt((box[2] - box[0]) * w * (box[3] - box[1]) * h) < 8:
                #if np.max(boxes) < 1 and min((box[2] - box[0]) * w, (box[3] - box[1]) * h) < 5:
                continue
            boxes_.append(box)
            labels_.append(label)
        return np.array(boxes_), np.array(labels_)



        #Rectangle(img, gt_truth, wh=(300, 300), img_name=imgName)
        #category = np.expand_dims(category, axis=1)
        #return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(labels) 

def Rectangle(img, gt_box, wh, img_name):
    gt_box[:, 0] = gt_box[:, 0] * wh[1]
    gt_box[:, 1] = gt_box[:, 1] * wh[0]
    gt_box[:, 2] = gt_box[:, 2] * wh[1]
    gt_box[:, 3] = gt_box[:, 3] * wh[0]
    #print(gt_box[:, 0].shape)
    #print(wh)
    gt_box = gt_box.astype(int)
    print(gt_box)
    for boxes in gt_box:
        #print(boxes)
        left_up, right_bottom = (boxes[0], boxes[1]), (boxes[2], boxes[3])
        cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)
    cv2.imwrite(os.path.join('/home/yosunpeng/github/SSD/InputImages', img_name), img)
    print('one image down')


            




'''
modelName = 'train'
modelDict = { 'train': 0, 'val': 1, 'test': 2}
rootPath = '/home/yosunpeng/github/DetectionDataset/VisDrone'
dataFolders = os.listdir(rootPath).remove('zips')
txtFiles = os.listdir(os.path.join(rootPath, dataFolders[0], 'annotations'))

example = '/home/yosunpeng/github/DetectionDataset/VisDrone/VisDrone2019-DET-train/9999998_00240_d_0000194.txt'




for txt in txtFiles:
    txtPath = os.listdir(os.path.join(rootPath, dataFolders[modelDict[modelName]], 'annotations', txt))
    with open(txtPath, 'r') as f:
        labels = f.readlines()

pdb.set_trace()
'''
