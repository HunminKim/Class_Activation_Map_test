import cv2
import numpy as np
import torch
import torchvision


def drow_heatmap(img, result, featuremap, parameter, height, width, overlap_heatmap=0.3):
    sel_params = parameter[result.argmax()]
    sel_params = sel_params.view(-1, 1, 1)
    _, w, h = featuremap.shape
    heatmap = torch.mul(sel_params.repeat((1, w, h)), featuremap)
    heatmap = torchvision.transforms.Resize((height, width))(heatmap)
    heatmap = heatmap.permute(1, 2, 0)
    heatmap = heatmap.sum(-1).detach().numpy()
    
    heatmap -= np.min(heatmap)
    heatmap /= np.max(heatmap)
    heatmap *= 255.
    heatmap = heatmap.astype(np.uint8).copy()
    
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_img = (heatmap * overlap_heatmap) + (img * (1-overlap_heatmap))
    return heatmap_img
