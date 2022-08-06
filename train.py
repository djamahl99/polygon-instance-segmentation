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
    return torch.sqrt(torch.sum((v1 - v2)**2))

def edge_loss(vertices, lengths):
    batch_total = 0

    for batch_num in range(vertices.shape[0]):
        b_verts = vertices[batch_num]
        batch_length = lengths[batch_num]
        total_dist = 0

        for edge_num in range(min(b_verts.shape[0], batch_length)):
            total_dist += edge_len(b_verts[edge_num - 1], b_verts[edge_num]) # negative okay, want to add distance from last -> first vertex 

        ave_dist = total_dist / b_verts.shape[0]

        total_variance = 0
        for edge_num in range(min(b_verts.shape[0], batch_length)):
            total_variance += torch.square(edge_len(b_verts[edge_num - 1], b_verts[edge_num]) - ave_dist) 

        batch_total += torch.sqrt(total_variance / min(b_verts.shape[0], batch_length))

    return batch_total

def perimeter_length_loss(pred_vertices, true_vertices, lengths):
    batch_total = 0
    batch_n = 0

    for batch_num in range(pred_vertices.shape[0]):
        pred_verts = pred_vertices[batch_num]
        true_verts = true_vertices[batch_num]
        batch_length = lengths[batch_num]
        batch_n += batch_length

        total_dist_pred = 0
        total_dist_true = 0

        for edge_num in range(min(pred_verts.shape[0], batch_length)):
            total_dist_pred += edge_len(pred_verts[edge_num - 1], pred_verts[edge_num]) # negative okay, want to add distance from last -> first vertex 
            total_dist_true += edge_len(true_verts[edge_num - 1], true_verts[edge_num]) # negative okay, want to add distance from last -> first vertex 

        batch_total += (total_dist_pred - total_dist_true) ** 2

    return batch_total / batch_n

def main():
    batch_size = 16
    epochs = 100
    load_model = True

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
    writer.add_scalar("Epochs", epochs)

    if os.path.exists(model_save_name) and load_model:
        model = torch.load(model_save_name).to(device)

    model.train(True)

    ####################################################################################
    # optimizers and lr scheduling ######################################################

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-8, momentum=0.9)

    criterion_chamfer = ChamferDistance().to(device)
    criterion_mse = nn.MSELoss().to(device)

    global_step = 0

    for epoch in trange(epochs):     
        n = 0
        running_loss = 0
        running_chamfer = 0
        running_edge = 0
        running_perimeter = 0
        running_mse = 0
        
        with tqdm(desc=f"EPOCH {epoch}", unit='img') as pbar:
            for in_imgs, true_vertices, length, params, contours, masks in tqdm(train_loader, unit='batch'):
                in_imgs = in_imgs.to(device=device, dtype=torch.float32)
                true_vertices = true_vertices.to(device=device, dtype=torch.float32).reshape(in_imgs.shape[0], -1, 2)
                params = params.to(device, dtype=torch.float32)
                contours = contours.to(device)
                masks = masks.to(device)
    
                pred_vertices = model(in_imgs, contours=contours, mask=masks)

                # plot_mask(in_imgs)

                loss_chamfer = criterion_chamfer(pred_vertices.reshape(in_imgs.shape[0], -1, 2), true_vertices.reshape(in_imgs.shape[0], -1, 2))
                loss_edges = edge_loss(pred_vertices, length)
                loss_perimeter = perimeter_length_loss(pred_vertices, true_vertices, length)
                # loss_mse = criterion_mse(pred_vertices, true_vertices)

                loss = loss_chamfer + loss_edges + loss_perimeter# + loss_mse

                running_loss += loss.item()
                running_chamfer += loss_chamfer.item()
                running_edge += loss_edges.item()
                running_perimeter += loss_perimeter.item()
                running_mse += 0#loss_mse.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                n += 1
                global_step += 1

                pbar.update(in_imgs.shape[0])

                pbar.set_postfix(**{'loss(ave)': running_loss/n, 'loss(chamfer)': running_chamfer/n, 'loss(edge)': running_edge/n, 'loss(perimeter)': running_perimeter/n, 'loss(mse-sample)': running_mse/n})

        # save_epoch_images(epoch, in_imgs, pred_len, pred_pos, true_pos, true_len, pred_angle=None, true_angle=None)
        torch.save(model, model_save_name)
        writer.flush()

    writer.close()

if __name__ == "__main__":
    main()