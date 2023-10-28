import datetime
import argparse
import pickle
import os
from pathlib import Path

import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from PIL import Image
import torch
import torchvision.transforms as transforms
import AdaIN_net as net
import custom_dataset as custData
import time as t


def get_losses_for_decoder(n_batches, optimizer, model, content_loader, style_loader, device, starting_batch = 0):

    model.train()
    total_loss = 0.0
    content_loss = 0.0
    style_loss = 0.0

    for batch in range(starting_batch, 1):
        t_3 = t.time()

        content = next(iter(content_loader)).to(device=device)
        style = next(iter(style_loader)).to(device=device)

        loss_c, loss_s = model(content, style)
        tot_curr_loss = loss_c + loss_s

        optimizer.zero_grad()
        tot_curr_loss.backward()
        optimizer.step()

        total_loss += tot_curr_loss.item()
        content_loss += loss_c.item()
        style_loss += loss_s.item()
        print('Batch #{}/{}         Time: {}'.format(batch, n_batches, (t.time() - t_3)))

    return ((total_loss/len(content_loader)), (content_loss/len(content_loader)), (style_loss/len(content_loader)))


if __name__ == '__main__':


    image_size = 512
    device = 'cpu'

    parser = argparse.ArgumentParser()

    parser.add_argument('-content_dir', '--content_dir', type=str, required=True,
                        help='Directory path to a batch of content images')
    parser.add_argument('-style_dir', '--style_dir', type=str, required=True,
                        help='Directory path to a batch of style images')

    # training options
    parser.add_argument('-gamma', '--gamma', default=1.0,
                        help='Gamma value')
    parser.add_argument('-e', '--e', type=int, default=50)
    parser.add_argument('-b', '--b', type=int, default=8)
    parser.add_argument('-l', '--l', help="Encoder.pth", required=True)
    parser.add_argument('-s', '--s', help="Decoder.pth")
    parser.add_argument('-p', '--p', help="decoder.png")
    parser.add_argument('-d', '--d', help="decoder.png")

    parser.add_argument('-cuda', '--cuda', default='Y')
    # python3 train.py
    # 	-content_dir. /../../../ datasets / COCO100 /
    # 	-style_dir. /../../../ datasets / wikiart100 /
    # 	-gamma 1.0
    # 	-e 20
    # 	-b 20
    # 	-l encoder.pth
    # 	-s decoder.pth
    # 	-p decoder.png
    # 	-cuda Y
    torch.cuda.empty_cache()

    args = parser.parse_args()


    # Import the dataset
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((480, 640))])

    content_data = custData.custom_dataset(args.content_dir, train_transform)
    style_data = custData.custom_dataset(args.style_dir, train_transform)

    num_batches = int(len(content_data) / args.b)

    content_data = DataLoader(content_data, args.b, shuffle=True)
    style_data = DataLoader(style_data, args.b, shuffle=True)

    # Pass to training
    # content_iter = iter(content_data)
    # style_iter = iter(style_data)

    # Set the device (GPU if available, otherwise CPU)
    print("Cuddda: {}     {}".format(torch.cuda.is_available(), torch.cuda.device_count()))
    if torch.cuda.is_available() and args.cuda == 'Y':
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("DEVICE USED: {}".format(device))
    # Create autoencoder
    decoder = net.encoder_decoder.decoder
    encoder = net.encoder_decoder.encoder
    encoder.load_state_dict(torch.load(args.l))

    for i in range(1, 21):
        decoder.load_state_dict(torch.load(('./10k_decoders/' + str(i) + '_decoder10k.pth'), map_location='cpu'))
        print("Loaded decoder:    {} ".format('./10k_decoders/' + str(i) + '_decoder10k.pth'))

        adain_model = net.AdaIN_net(encoder, decoder)
        adain_model.to(device)
        # Define optimizer and learning rate scheduler
        optimizer = optim.Adam(adain_model.parameters(), lr=0.0005, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True, factor=0.1,
                                                         min_lr=1e-4)
    # Train the model

        loss = get_losses_for_decoder(num_batches, optimizer, adain_model, content_data, style_data, device)
        print("The avg losses for current decoder is: {}".format(loss))
