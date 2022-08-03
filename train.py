from cmath import isnan
from glob import glob
import json
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import trange, tqdm
import natsort
import wandb
import logging

import cv2
from chamferdist import ChamferDistance

import pytorch_warmup as warmup

from modules.PolyTransform import PolyTransform

from torch.utils.tensorboard import SummaryWriter

from datasets.coco_pts import COCO_PTS
from visualize.plot_positions import plot_mask
writer = SummaryWriter()

def edge_len(v1, v2):
    # print("edge len", v1, v2)
    return torch.sqrt(torch.sum((v1 - v2)**2))

def edge_loss(vertices):
    batch_total = 0

    # print("edge loss shape", vertices.shape)
    for batch_num in range(vertices.shape[0]):
        b_verts = vertices[batch_num]
        total_dist = 0
        for edge_num in range(b_verts.shape[0]):
            # print(f"cdist verts for {edge_num} - ", b_verts[edge_num - 1], b_verts[edge_num])
            total_dist += edge_len(b_verts[edge_num - 1], b_verts[edge_num]) # negative okay, want to add distance from last -> first vertex 

        ave_dist = total_dist / b_verts.shape[0]

        total_variance = 0
        for edge_num in range(b_verts.shape[0]):
            total_variance += torch.square(edge_len(b_verts[edge_num - 1], b_verts[edge_num]) - ave_dist) 

        batch_total += torch.sqrt(total_variance / b_verts.shape[0])

    return batch_total

def main():
    batch_size = 16
    epochs = 20
    lr = 1e-5
    load_model = False

    ####################################################################################
    # load dataset #####################################################################
    # dataset = COCO_PTS('key_pts/key_pts_instances_train2017.json')
    dataset = COCO_PTS('key_pts/key_pts_instances_val2017.json')
    val_dataset = COCO_PTS('key_pts/key_pts_instances_val2017.json')

    train_loader = torch.utils.data.DataLoader(dataset,
                                        batch_size = batch_size,
                                        num_workers = 4,
                                        shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                        batch_size = batch_size,
                                        num_workers = 0,
                                        shuffle = True)

    amp = False
    device = torch.device('cuda')

    model = PolyTransform().to(device)

    model_save_name = f"{model._get_name()}.pt"
    writer.add_text("Model Name", model._get_name())
    writer.add_scalar("Learning Rate", lr)
    writer.add_scalar("Epochs", epochs)

    if os.path.exists(model_save_name) and load_model:
        model = torch.load(model_save_name).to(device)

    model.train(True)

    ####################################################################################
    # optimizers and lr scheduling ######################################################

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    # scheduler = torch.optim.lr_scheduler.St
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    criterion_chamfer = ChamferDistance().to(device)

    global_step = 0

    for epoch in trange(epochs):     
        n = 0
        running_loss = 0
        running_chamfer = 0
        running_edge = 0
        
        with tqdm(desc=f"EPOCH {epoch}", unit='img') as pbar:
            for in_imgs, true_vertices in tqdm(train_loader, unit='batch'):
                in_imgs = in_imgs.to(device=device, dtype=torch.float32)
                true_vertices = true_vertices.to(device=device, dtype=torch.float32)
                
                pred_vertices = model(in_imgs)

                # plot_mask(in_imgs)

                loss_chamfer = criterion_chamfer(pred_vertices.reshape(in_imgs.shape[0], -1, 2), true_vertices.reshape(in_imgs.shape[0], -1, 2))
                loss_edges = edge_loss(pred_vertices)

                loss = loss_chamfer + loss_edges

                running_loss += loss.item()
                running_chamfer += loss_chamfer.item()
                running_edge += loss_edges.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                n += 1
                global_step += 1

                pbar.update(in_imgs.shape[0])

                pbar.set_postfix(**{'loss(ave)': running_loss/n, 'loss(chamfer)': running_chamfer/n, 'loss(edge)': running_edge/n})

        # save_epoch_images(epoch, in_imgs, pred_len, pred_pos, true_pos, true_len, pred_angle=None, true_angle=None)
        torch.save(model, model_save_name)
        writer.flush()

    writer.close()

if __name__ == "__main__":
    main()