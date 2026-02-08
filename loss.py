from torchvision import transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
import pydiffvg
from utils.image_process import *
from torchvision.transforms import ToPILImage


def weights_mse(mask,raster_img,target_img,scale=1):
    loss_mse = F.mse_loss(raster_img, target_img)
    mask_raster_img = mask*raster_img
    mask_target_img = mask*target_img
    loss_mse += F.mse_loss(mask_raster_img, mask_target_img)*10000
    return loss_mse*scale


def exclude_loss(raster_img,scale=1):
    img = F.relu(178/255 - raster_img)
    loss = torch.sum(img)*scale
    return loss

def weight_loss(size_tensor,target_area_list,scale):
    size_tensor1 = size_tensor.squeeze()
    loss = 1-F.cosine_similarity(size_tensor1, target_area_list, dim=0)
    return loss*scale

def force_loss(pos_tensor,gravity_direction,scale=1,img_size=1000):
    x1 = pos_tensor.squeeze()
    if gravity_direction=="left":
        result = x1[:,0]
    if gravity_direction == "down":
        result = img_size-x1[:,1]
    if gravity_direction == "point":
        result = torch.abs(400-x1[:,0])+torch.abs(400-x1[:,1])
    loss = torch.sum(result)
    loss = loss*scale
    return loss

def uniform_loss(mask1,raster_img,diflation_list,scale=1):
    loss = 0
    raster_img = rgb_to_grayscale(raster_img)
    count = 0
    for model in diflation_list:
        count+=1
        output_image = model(raster_img)*mask1
        loss+=torch.sum(output_image)
        image = ToPILImage()(output_image.detach())

    return loss*scale
