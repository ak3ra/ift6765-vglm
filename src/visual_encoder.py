import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np

def get_roi_pooled_features(object_detector,image_input,bnbox,pool_width=7,pool_height=7,device='cpu'):
    """
    image_input, a torch tensor reprenting a single image, [batch,channel,height, width]
    bnbox, bounding boxes of an image region: a torch tensor in absolute pixel numbers, [number_boxes,4] 
    """

    # model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    model = object_detector
    model.to(device)
    model.eval()

    visual_features_before_roi = []
    hook = model.backbone.register_forward_hook(
        lambda self, input, output: visual_features_before_roi.append(output))
    res = model(image_input)# not useful
    hook.remove()

    m = torchvision.ops.MultiScaleRoIAlign(['0', '1','2','3'], pool_width, pool_height)
    # original image size, before computing the feature maps
    image_sizes = [(image_input.shape[2], image_input.shape[3])]
    pooled_feat = m(visual_features_before_roi[0], [bnbox], image_sizes) # [7,7,256]
    pooled_feat = pooled_feat.flatten(start_dim=1)
    pooled_feat = model.roi_heads.box_head.fc6(pooled_feat).detach()
    pooled_feat= model.roi_heads.box_head.fc7(pooled_feat).detach()

    return pooled_feat

if __name__=="__main__":
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    boxes = torch.rand(2, 4) * 200; boxes[:, 2:] += boxes[:, :2]
    image= torch.rand(1,3,224,224)
    pooled_feat = get_roi_pooled_features(model,image,boxes)
    assert pooled_feat.shape==(len(boxes),1024)


