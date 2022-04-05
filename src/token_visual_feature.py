import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
from tqdm import tqdm

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


import pickle
from PIL import Image
from torchvision import transforms
import numpy as np


image_num_chosen = 5 # randomly selected 5 corresponding images
voken_dim = 1024
token_2_image_filename="token_2_image_flick30k.p"
token_2_image_flickr30k = pickle.load( open( token_2_image_filename, "rb" ) )
token_list_filename='../dataset/flickr30/flickr30k_class_name.txt'
images_path = '/Volumes/Grace-Mac/Vision_Language/flickr30/flickr30k_images/flickr30k_images/'# folder of flicrk30k images
model = fasterrcnn_resnet50_fpn(pretrained=True)

output_filename='token_2_feature_flick30k.p'

convert_tensor = transforms.ToTensor()



with open(token_list_filename,'r') as f:
  lines=f.readlines()
token_visual_feature = {}
tokens_list =[e.strip() for e in lines]


for i in tqdm(range(len(tokens_list))):
  token = tokens_list[i]
# for token in tokens_list[:2]:
  # print(token)
  token_2_image_list = token_2_image_flickr30k[token]
  index_sels = np.random.choice(np.arange(len(token_2_image_list)),image_num_chosen,replace=False)
  pooled_feat = torch.zeros((1,voken_dim))

  for index_sel in index_sels:
    image_name = token_2_image_flickr30k[token][index_sel][0]
    bnbox = token_2_image_flickr30k[token][index_sel][1]
    bnbox = torch.unsqueeze(bnbox,0) # add a batch dimension
    # print(token_2_image_flickr30k[token][index_sel])

    img = Image.open(images_path+image_name)
    img_tensor=convert_tensor(img)
    img_tensor = torch.unsqueeze(img_tensor,0)

    pooled_feat += get_roi_pooled_features(model,img_tensor,bnbox)

  
  token_visual_feature[token] = pooled_feat/image_num_chosen # calculate the average

pickle.dump(token_visual_feature , open( output_filename, "wb" ) )
# token_2_feature_flickr30k = pickle.load( open( "token_2_feature_flick30k.p", "rb" ) )
