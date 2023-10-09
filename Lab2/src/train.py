import datetime
import argparse
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


# class DatasetLoader(Dataset):
#     def __init__(self, img_dir, transform):
#         super().__init__()
#
#         if not os.path.exists(img_dir):
#             raise ValueError("This directory does not exist")
#         self.img_dir = img_dir
#         self.transform = transform
#
#         self.imgs = list(Path(self.img_dir).glob('*'))
#
#     def __getitem__(self, item):
#         try:
#             img = Image.open(self.imgs[item]).convert('RGB')
#             img = self.transform(img)
#             return img
#         except Exception as e:
#             print("Caught error in loading image w index {0}\nError: {1}".format(item, e))
#             return None
#
#     def __len__(self):
#         return len(self.imgs)


def train(n_epochs, n_batches, optimizer, model, content_loader, style_loader, scheduler, device,
          decoder_save=None, alpha=1.0, plot_file=None):
    print("Training started")
    model.train()
    total_losses = []
    content_losses = []
    style_losses = []
    final_loss = 0.0
    t_1 = t.time()

    for epoch in range(1, n_epochs + 1):
        print("Epoch", epoch)
        # Losses for current epoch
        total_loss = 0.0
        content_loss = 0.0
        style_loss = 0.0
        t_2 = t.time()

        for batch in range(n_batches):
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

        if decoder_save is not None:
            torch.save(model.state_dict(), decoder_save)

        scheduler.step(total_loss)
        total_losses.append(total_loss / len(content_loader))
        content_losses.append(content_loss / len(content_loader))
        style_losses.append(style_loss / len(style_loader))

        final_loss = total_loss / len(content_loader)
        print('Epoch {}, Training loss {}, Time  {}'.format(epoch, total_loss / len(content_loader),(t.time() - t_1)))

    # summary(model, (1, 28 * 28))
    return final_loss

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
    parser.add_argument('-l', '--l', help="Encoder.pth")
    parser.add_argument('-s', '--s', help="Decoder.pth")
    parser.add_argument('-p', '--p', help="decoder.png")
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

    num_batches = len(content_data) / args.b

    content_data = DataLoader(content_data, args.b, shuffle=True)
    style_data = DataLoader(style_data, args.b, shuffle=True)

    # Pass to training
    # content_iter = iter(content_data)
    # style_iter = iter(style_data)

    # Set the device (GPU if available, otherwise CPU)

    if torch.cuda.is_available() and args.cuda == 'Y':
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    # Create autoencoder
    adain_model = net.AdaIN_net(net.encoder_decoder.encoder, net.encoder_decoder.decoder)
    adain_model.encoder.load_state_dict(torch.load(args.l))

    adain_model.to(device)
    # Define optimizer and learning rate scheduler
    optimizer = optim.Adam(adain_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True, factor=0.1,
                                                     min_lr=1e-4)
    # Train the model
    train(args.e, num_batches, optimizer, adain_model, content_data, style_data, scheduler, device, args.s, 1.0, args.p)
