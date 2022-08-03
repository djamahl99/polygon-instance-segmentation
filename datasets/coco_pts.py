from cmath import isnan
import enum
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch 

import json 
import os
import numpy as np
import cv2

MAX_SEQ_LEN = 50

class COCO_PTS(Dataset):
    def __init__(self, key_pts):
        self.key_pts = json.loads(open(key_pts).read()) 
        self.idxs = list(self.key_pts.keys())
        self.buffer = 5 # edge buffer

        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        key_pts_list = np.array(self.key_pts[self.idxs[idx]])

        key_pts_list = key_pts_list.reshape(-1, 2)

        in_img = np.zeros((224, 224))

        for i in range(key_pts_list.shape[0]):
            key_pt = key_pts_list[i]
            key_pts_list[i] = [np.clip(key_pt[0], self.buffer, 223 - self.buffer), np.clip(key_pt[1], self.buffer, 223 - self.buffer)]

        # repeat last element to fit 50 vertices
        for i in range(abs(50 - key_pts_list.shape[0])):
            key_pts_list = np.append(key_pts_list, key_pts_list[-1, :].reshape(1, 2), axis=0)

        key_pts_list = key_pts_list.reshape(1, -1, 2)

        in_img = cv2.fillPoly(in_img, key_pts_list, color=(255,255,255))

        return self.input_transform(in_img), key_pts_list