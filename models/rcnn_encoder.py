import torch
import torchvision
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import AnchorGenerator

device = torch.device("cuda")


def load_model():
    resnext = torchvision.models.resnext50_32x4d(pretrained=True)
    modules = list(resnext.children())[:-1]
    backbone = nn.Sequential(*modules)

    backbone.out_channels = 2048
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    model = FasterRCNN(backbone,
                   num_channels=4,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

    return model.to(device).eval()

def call_model(data):
    predictions = load_model(data)

    return predictions


