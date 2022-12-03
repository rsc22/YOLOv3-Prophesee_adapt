from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from plot_utils import plot_losses, plot_img
import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


class ColoringYOLO(nn.Module):
    """Custom architecture adding one convolution before the Darknet so that
    it translates the black and white input into a colored image as expected
    by a Darknet trained on KITTI"""
    def __init__(self, darnket, img_size=416):
        super(ColoringYOLO, self).__init__()
        self.conv_color = nn.Conv2d(1,3,5, padding='same', padding_mode='replicate')
        self.darknet = darnket

    def forward(self, x, targets=None):
        x = self.conv_color(x)
        total_loss = self.darknet(x, targets)
        self.update_attributes()
        return total_loss

    def update_attributes(self):
        self.losses = self.darknet.losses
        self.seen = self.darknet.seen

def load_color_layer(path='weights/best_color.weights'):
    layer = torch.load(path, map_location=torch.device('cpu'))
    return layer

def plot_colored_images(weights_path):
    test_path = r'F:\Documentos\Data_Science\Master\UOC_Ingenieria_Computacional_y_Matematica\Master_UOC\TFM\YOLO\PyTorch-YOLOv3-kitti\data\proph\train'
    cuda = torch.cuda.is_available()

    data_config = parse_data_config("config/kitti.data")
    test_path = data_config["train"]
    dataset = ListDataset(test_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    #img = np.array(Image.open(im_path).convert("RGB"))
    convlayer = load_color_layer(weights_path)
    print(convlayer)
    if cuda:
        model = convlayer.cuda()
        model.eval()
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        imgs = Variable(imgs.type(Tensor))
        # channel_equality = np.array_equal(imgs.cpu()[0,0,...], imgs.cpu()[0,1,...])
        # print(f"All channels are equal:{channel_equality} ")
        orig_imgs = imgs.cpu().numpy().copy()
        imgs = imgs[:,:1,...]
        with torch.no_grad():
            imgs = model(imgs)

        plot_img(imgs)

if __name__ == '__main__':
    # darknet = Darknet("config/yolov3-kitti.cfg")
    # model = ColoringYOLO(darknet)
    # print(model.conv_color)

    plot_colored_images("weights/best_color_1c.weights")
