import os
import argparse
import numpy as np
import cv2
import torch

from util import drow_heatmap
from model import ClassificationModel

def define_argparser():
    parser = argparse.ArgumentParser('Class Activation Map')
    parser.add_argument('--input_img', default='./test.jpg')
    parser.add_argument('--save_img', default='./heatmap.jpg')
    parser.add_argument('--input_size', default=224)
    return parser.parse_args()

def main():
    args = define_argparser()
    model = ClassificationModel()
    model.eval()
    parameter = model.fc_parameters
    img = cv2.imread(args.input_img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if isinstance(img, type(None)):
        print('no image')
        exit()
    height, width, _ = np.shape(img)

    img_resized = cv2.resize(img_rgb, (args.input_size, args.input_size))
    img_input = torch.from_numpy(img_resized).permute(2, 0, 1) / 255.
    img_input = torch.unsqueeze(img_input, 0).type(torch.float32)
    result = model(img_input)
    result = torch.softmax(result, -1)
    print(result.argmax())
    featuremap = model.featuremap[0]

    heatmap_img = drow_heatmap(img, result, featuremap, parameter, height, width)
    cv2.imwrite(args.save_img, heatmap_img)

if __name__ == '__main__':
    main()