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

        length = key_pts_list.shape[0]

        in_img = np.zeros((224, 224))

        for i in range(key_pts_list.shape[0]):
            key_pt = key_pts_list[i]
            key_pts_list[i] = [np.clip(key_pt[0], self.buffer, 223 - self.buffer), np.clip(key_pt[1], self.buffer, 223 - self.buffer)]

        # repeat last element to fit 50 vertices
        for i in range(abs(50 - key_pts_list.shape[0])):
            key_pts_list = np.append(key_pts_list, key_pts_list[-1, :].reshape(1, 2), axis=0)

        key_pts_list = key_pts_list.reshape(1, -1, 2)

        in_img = cv2.fillPoly(in_img, key_pts_list, color=(255,255,255))

        # params 
        center = np.mean(key_pts_list.reshape(-1, 2), axis=0)
        radius = 0
        for j in range(50):
            d = np.sqrt(np.sum((key_pts_list[0][j] - center) ** 2))
            if d > radius:
                radius = d

        params = torch.tensor([center[0], center[1], radius])

        # contours #############
        mask_in = np.array(in_img, dtype=np.uint8)
        ret, thresh = cv2.threshold(mask_in, 1, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # print(contours)
        contour = contours[0].reshape(-1, 2)
    
        # print("shape before", contour.shape)

        if len(contour) > 50:
            # print("too long")
            perimeter = cv2.arcLength(contour, True)
            # print('perimeter is', perimeter)
            px_diff = perimeter / 50

            contour_ = [contour[0]]
            last = contour[0]
            idx = 1

            while len(contour_) < 50 and idx < len(contour):
                if np.linalg.norm(last - contour[idx]) >= 0.95 * px_diff:
                    contour_.append(contour[idx])
                    last = contour[idx]

                idx += 1

            contour = np.array(contour_)


        # select every x amount to get 50 
        # step = contour.shape[0] // 50
        # step = max(step, 1)
        # contour = contour[::step, :] # maybe change
        # if contour.shape[0] > 50:
            # contour = contour[[min(i*step, contour.shape[0] - 1) for i in range(50)], :]
        # print("shape after selection", contour.shape)

        # repeat last element to fit 50 vertices 
        for i in range(abs(50 - contour.shape[0])):
            # print("appending...")
            contour = np.append(contour, contour[-1, :].reshape(1, 2), axis=0)
        # contours = contours[0:50, :] # maybe fix
        del mask_in, thresh, ret, hierarchy

        contour = torch.tensor(contour, dtype=torch.float32)

        mask = [1 if i < length else 0 for i in range(50)]
        mask = torch.tensor(mask).bool()

        return self.input_transform(in_img), key_pts_list, length, params, contour, mask