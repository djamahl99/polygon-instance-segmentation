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
    def __init__(self, num_vertices=50) -> None:
        super().__init__()

        self.w_c = 224
        self.h_c = 224
        self.num_vertices = num_vertices

        # grayscale to rgb 1x1 conv
        self.gray_to_rgb = nn.Conv2d(1, 3, 1)

        # resnet fpn
        self.fpn = resnet()
        self.fpn.create_architecture()

        # channel reduction 256 -> 64
        self.fpn_reduction_p2 = fpn_reduction()
        self.fpn_reduction_p3 = fpn_reduction()
        self.fpn_reduction_p4 = fpn_reduction()
        self.fpn_reduction_p5 = fpn_reduction()
        self.fpn_reduction_p6 = fpn_reduction()

        # upsampling
        self.upsample = nn.Upsample(size=(self.h_c, self.w_c))

        # coordinate layers
        self.add_coords = AddCoords()

        # Transformer
        self.deformable_vertex_transformer = DeformableVertexTransformer(num_vertices=self.num_vertices, dim=1024, depth=4, heads=8, mlp_dim=512)

    def forward(self, input):
        if input.shape[1] == 1:
            # grayscale to rgb 1x1 conv
            input = self.gray_to_rgb(input)

        # ResNet101 FPN
        p2, p3, p4, p5, p6 = self.fpn(input)

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
        indices = torch.tensor(self._initialize_polygon(None), dtype=torch.float).reshape(1, 1, -1, 2).repeat(x.shape[0], 1, 1, 1).to(x.device)
        sample = F.grid_sample(x, indices).permute(0, 3, 2, 1).squeeze(2)

        # transformer
        vertices = self.deformable_vertex_transformer(sample)

        return vertices.reshape(-1, 50, 2)

    def _initialize_polygon(self, input):
        def circle(n):
            indices = []

            for i in range(n):
                x = np.cos(2*np.pi * (i / n))
                y = np.sin(2*np.pi * (i / n))

                indices.append([x, y])
            
            return indices

        return circle(50)

        # later use contour algorithm

if __name__ == "__main__":
    model = PolyTransform()

    image = torch.zeros((2, 3, 224, 224))
    o = model(image)
    print(o.shape)