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


def train(n_epochs, n_batches, optimizer, model, content_loader, style_loader, scheduler, device,
          decoder_save=None, gamma=1.0, plot_file=None, pickleLosses = None, starting_epoch = 1, starting_batch = 0):

    model.train()
    total_losses = []
    content_losses = []
    style_losses = []
    final_loss = 0.0
    t_1 = t.time()

    # loading_saved pickle data
    if pickleLosses is not None:
        try:
            with open(pickleLosses, 'rb') as file:
                loaded_losses = pickle.load(file)
                (total_losses, content_losses, style_losses) = loaded_losses
                print("Loaded saved losses from file successfully: \n{} \n{} \n{}".format(total_losses, content_losses, style_losses))
        except Exception as e:
            print(f"An error occurred while loading arrays: {str(e)}")

    print("Training started at Epoch {}".format(starting_epoch))


    for epoch in range(starting_epoch, n_epochs + 1):
        print("Epoch", epoch)
        # Losses for current epoch
        total_loss = 0.0
        content_loss = 0.0
        style_loss = 0.0
        t_2 = t.time()

        for batch in range(starting_batch, n_batches):
            t_3 = t.time()

            content = next(iter(content_loader)).to(device=device)
            style = next(iter(style_loader)).to(device=device)

            loss_c, loss_s = model(content, style)
            tot_curr_loss = loss_c + (gamma * loss_s)

            optimizer.zero_grad()
            tot_curr_loss.backward()
            optimizer.step()

            total_loss += tot_curr_loss.item()
            content_loss += loss_c.item()
            style_loss += loss_s.item()
            print('Batch #{}/{}         Time: {}'.format(batch, n_batches, (t.time() - t_3)))

        if decoder_save is not None:
            torch.save(model.decoder.state_dict(), (str(epoch) + '_' + decoder_save))
            print("Saved decoder model under name: {}".format(str(epoch) + '_' + decoder_save))

        scheduler.step(total_loss)
        total_losses.append(total_loss / len(content_loader))
        content_losses.append(content_loss / len(content_loader))
        style_losses.append(style_loss / len(style_loader))

        final_loss = total_loss / len(content_loader)
        pickleSave = "pickled_dump.pk1"
        if pickleLosses is not None:
            pickleSave = pickleLosses

        try:
            with open(pickleSave, 'wb') as file:
                pickle.dump((total_losses, content_losses, style_losses), file)
                print("Saved losses to '{}'".format(pickleSave))
        except Exception as e:
            print(f"An error occurred while saving arrays: {str(e)}")

        if plot_file is not None:
            plt.figure(2, figsize=(12, 7))
            plt.clf()
            plt.plot(total_losses, label='Total')
            plt.plot(style_losses, label='Style')
            plt.plot(content_losses, label='Content')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc=1)
            plt.savefig(plot_file)

        print('Epoch {}, Training loss {}, Time  {}'.format(epoch, total_loss / len(content_loader),(t.time() - t_2)))

    print('Total Training loss {}, Time  {}'.format(final_loss, (t.time() - t_1)))

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
    parser.add_argument('-l', '--l', help="Encoder.pth", required=True)
    parser.add_argument('-s', '--s', help="Decoder.pth")
    parser.add_argument('-p', '--p', help="decoder.png")
    parser.add_argument('-starting_epoch', '--starting_epoch', type=int, help="3", default=1)
    parser.add_argument('-starting_decoder', '--starting_decoder', help="#_decoder.pth")
    parser.add_argument('-starting_pickle', '--starting_pickle', help="pickledLosses.pk1", default=None)

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

    if args.starting_decoder is not None:
        decoder.load_state_dict(torch.load(args.starting_decoder))
        # adain_model = net.AdaIN_net(net.encoder_decoder.encoder, net.encoder_decoder.decoder)
        # adain_model.load_state_dict((torch.load(args.starting_decoder)))

    adain_model = net.AdaIN_net(encoder, decoder)
    adain_model.to(device)
    # Define optimizer and learning rate scheduler
    optimizer = optim.Adam(adain_model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True, factor=0.1,
                                                     min_lr=1e-4)
    # Train the model
    train(args.e, num_batches, optimizer, adain_model, content_data, style_data, scheduler, device, args.s, args.gamma,
          args.p, starting_epoch=args.starting_epoch, pickleLosses=args.starting_pickle)
