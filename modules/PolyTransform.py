import os, sys
# sys.path.append(os.path.join(os.path.dirname(__file__), "fpn.pytorch/lib/"))
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch_dct
import numpy as np
from modules.DeformableVertexTransformer import DeformableVertexTransformer

from modules.resnet import resnet
from layers.CoordConv import AddCoords
from perceiver_pytorch import Perceiver

def fpn_reduction():
    """
        "[...][W]e take the P2, P3, P4, P5, and P6 feature maps and apply two lateral convolutional layers 
        to each of them in order to reduce the number of feature channels from 256 to 64 each."   
    """
    return nn.Sequential(
        nn.Conv2d(256, 64, 1), 
        nn.ReLU(True),
        nn.Conv2d(64, 64, 1),
        nn.ReLU(True),
    )

class PolyTransform(nn.Module):
    def __init__(self, num_vertices=50, use_unet=False, use_contours=True) -> None:
        super().__init__()

        self.w_c = 224
        self.h_c = 224
        self.num_vertices = num_vertices
        self.use_unet = use_unet
        self.use_contours = use_contours

        # grayscale to rgb 1x1 conv
        self.gray_to_rgb = nn.Conv2d(1, 3, 1)

        # resnet fpn
        self.fpn = resnet()
        self.fpn.create_architecture()

        if self.use_unet and not self.use_contours:
            # resnet unet vertex extraction
            self.unet = torch.load("ResNetUNet.pt")
        elif not self.use_contours:
            # learnable sampling indices
            self.sampling_mlp = nn.Sequential(
                nn.Linear(256 * 7 * 7, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 3),
                nn.ReLU(True)
            )

        # channel reduction 256 -> 64
        self.fpn_reduction_p2 = fpn_reduction()
        self.fpn_reduction_p3 = fpn_reduction()
        self.fpn_reduction_p4 = fpn_reduction()
        self.fpn_reduction_p5 = fpn_reduction()
        self.fpn_reduction_p6 = fpn_reduction()

        # upsampling
        self.upsample = nn.Upsample(size=(self.h_c, self.w_c), mode='bilinear', align_corners=True)

        # coordinate layers
        self.add_coords = AddCoords()

        # Transformer
        self.perceiver = DeformableVertexTransformer(num_vertices=self.num_vertices, dim=1024, depth=4, heads=8, mlp_dim=512)
        # self.perceiver = Perceiver(
        #     input_channels = 322,          # number of channels for each token of the input
        #     input_axis = 1,              # number of axis for input data (2 for images, 3 for video)
        #     num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
        #     max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
        #     depth = 6,                   # depth of net. The shape of the final attention mechanism will be:
        #                                 #   depth * (cross attention -> self_per_cross_attn * self attention)
        #     num_latents = 50,           # number of latents, or induced set points, or centroids. different papers giving it different names
        #     latent_dim = 512,            # latent dimension
        #     cross_heads = 1,             # number of heads for cross attention. paper said 1
        #     latent_heads = 8,            # number of heads for latent self attention, 8
        #     cross_dim_head = 128,         # number of dimensions per cross attention head
        #     latent_dim_head = 128,        # number of dimensions per latent self attention head
        #     num_classes = 50*2,          # output number of classes
        #     attn_dropout = 0.1,
        #     ff_dropout = 0.1,
        #     weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
        #     fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
        #     self_per_cross_attn = 2      # number of self attention blocks per cross attention
        # )

        self.out = nn.Sequential(
            nn.Linear(50*322, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 50*2),
        )

    def forward(self, input, contours=None, mask=None):
        if input.shape[1] == 1:
            # grayscale to rgb 1x1 conv
            input_rgb = self.gray_to_rgb(input)

        # ResNet101 FPN
        p2, p3, p4, p5, p6 = self.fpn(input_rgb)

        if self.use_contours:
            # scale to [-1,1] range from [0,224]
            contours_grid = contours / 112 - 1
            # print("min,max", contours.min(), contours.max())

        elif self.use_unet:
            # ResNetUNet vertex extraction
            _, vert_probs = self.unet(input)
            indices_ = []
            for b in range(input.shape[0]):
                _, indices = torch.topk(vert_probs[b].flatten(0), self.num_vertices)
                print("top k indices", indices.shape)
                indices = (np.array(np.unravel_index(indices.detach().cpu().numpy(), input.shape[1:])))
                print("indices shape", indices.shape)
                indices_.append(indices)

            indices = torch.tensor(indices_)
            print("final indices shape", indices.shape)
            # indices = indices.reshape(-1, 2)
        else:
            # learnable sampling indices
            sample_parameters = self.sampling_mlp(p6.flatten(start_dim=1))

        # Reduce channels from 256 to 64
        p2 = self.fpn_reduction_p2(p2)
        p3 = self.fpn_reduction_p3(p3)
        p4 = self.fpn_reduction_p4(p4)
        p5 = self.fpn_reduction_p5(p5)
        p6 = self.fpn_reduction_p6(p6)

        # upsample to H_c x W_c x 64 for each 
        p2 = self.upsample(p2)
        p3 = self.upsample(p3)
        p4 = self.upsample(p4)
        p5 = self.upsample(p5)
        p6 = self.upsample(p6)

        # concatenate to H_c x W_c x (320)
        x = torch.cat([p2, p3, p4, p5, p6], dim=1)
        
        # add coordinates layer b x H_c x W_c x (320 + 2)
        x = self.add_coords(x)

        # sampling using polygon initialization -> N x (320 + 2)
        if self.use_contours:
            # some of this may be unnecessary
            indices = contours_grid.to(x.device, dtype=torch.float32).reshape(x.shape[0], 1, -1, 2)
        elif not self.use_unet:
            # indices = torch.tensor(self._initialize_polygon(sample_parameters), dtype=torch.float).reshape(1, 1, -1, 2).repeat(x.shape[0], 1, 1, 1).to(x.device)
            # indices = sample_indices.reshape(x.shape[0], 1, -1, 2)
            indices = torch.tensor(self._initialize_polygon(sample_parameters), dtype=torch.float).reshape(x.shape[0], 1, -1, 2).to(x.device)
        else:
            indices = torch.tensor(indices, dtype=torch.float).reshape(x.shape[0], 1, -1, 2).to(x.device)

        sample = F.grid_sample(x, indices, align_corners=True).permute(0, 3, 2, 1).squeeze(2)

        # transformer
        # vertices = self.deformable_vertex_transformer(sample)
        # vertices = self.perceiver(sample, mask=mask).reshape(-1, 50, 2) + contours
        vertices = self.perceiver(sample).reshape(-1, 50, 2) + contours

        # vertices = self.out(sample.flatten(start_dim=1)).reshape(-1, 50, 2) + contours

        return vertices

    def _initialize_polygon(self, parameters):
        def circle(parameters, n=50):
            indices = []

            for batch in range(parameters.shape[0]):
                batch_indices = []
                p = parameters[batch]

                for i in range(n):
                    x = p[2].item() * np.cos(2*np.pi * (i / n)) + p[0].item()
                    y = p[2].item() * np.sin(2*np.pi * (i / n)) + p[1].item()

                    batch_indices.append([x, y])
                
                indices.append(batch_indices)
            
            return indices

        return circle(parameters)

        # later use contour algorithm

if __name__ == "__main__":
    model = PolyTransform()

    image = torch.zeros((2, 3, 224, 224))
    o = model(image)
    print(o.shape)