
import torchviz
from torchviz import make_dot
import argparse
import pickle
import ssl
import os

import torch.optim as optim
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import time as t

import vanilla_model as vanilla
import moded_model as moded
if __name__ == '__main__':
    device = 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--l', help="Encoder.pth", required=True)
    parser.add_argument('-s', '--s', help="Decoder.pth")
    args = parser.parse_args()

    encoder = vanilla.encoder_decoder.encoder
    encoder.load_state_dict(torch.load(args.l))
    frontend = vanilla.encoder_decoder.frontend
    model = vanilla.model(encoder, frontend)
    model.to(device)
    model.eval()
    tmpimg = (1,3,32,32)
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    cifar_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    cifar_data = DataLoader(cifar_dataset, batch_size=1024, shuffle=True)
    for imgs, label in cifar_data:

        with torch.no_grad():
            output = model(imgs)
            dot = make_dot(output, params=dict(list(model.named_parameters()) + [('x', imgs)]))

        # Save the visualization to a file
        dot.render('model_visualization', format='png')
        break
