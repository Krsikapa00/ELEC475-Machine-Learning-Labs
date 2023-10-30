import datetime
import argparse
import pickle
import os
from pathlib import Path

import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import vanilla_model as vanilla
import torch
import torchvision.transforms as transforms
import custom_dataset as custData
import time as t


def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device,
          decoder_save=None, plot_file=None, pickleLosses = None,
          starting_epoch = 1):

    model.train()
    total_losses = []
    final_loss = 0.0
    t_1 = t.time()

    # loading_saved pickle data
    if pickleLosses is not None:
        try:
            with open(pickleLosses, 'rb') as file:
                loaded_losses = pickle.load(file)
                (total_losses, content_losses, style_losses) = loaded_losses
                print("Loaded saved losses from file successfully: \n{} \n{} \n{}"
                      .format(total_losses, content_losses, style_losses))
        except Exception as e:
            print(f"An error occurred while loading arrays: {str(e)}")

    print("Training started at Epoch {}".format(starting_epoch))

    for epoch in range(starting_epoch, n_epochs + 1):
        print("Starting Epoch: ", epoch)
        # Losses for current epoch
        total_loss = 0.0

        t_2 = t.time()

        for idx, data in enumerate(train_loader):
            t_3 = t.time()
            imgs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = model(imgs)
            loss = loss_fn(outputs, imgs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print('Batch #{}/{}         Time: {}'.format(idx + 1, len(train_loader), (t.time() - t_3)))

        if decoder_save is not None:
            torch.save(model.decoder.state_dict(), (str(epoch) + '_' + decoder_save))
            print("Saved decoder model under name: {}".format(str(epoch) + '_' + decoder_save))

        scheduler.step(total_loss)
        total_losses.append(total_loss / len(train_loader))

        final_loss = total_loss / len(train_loader)

        # Store loss data in pickled file
        pickleSave = "pickled_dump.pk1"
        if pickleLosses is not None:
            pickleSave = pickleLosses
        try:
            with open(pickleSave, 'wb') as file:
                pickle.dump((total_losses, content_losses, style_losses), file)
                print("Saved losses to '{}'".format(pickleSave))
        except Exception as e:
            print(f"An error occurred while saving arrays: {str(e)}")

        # Plot loss data each epoch
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

        print('Epoch {}, Training loss {}, Time  {}'.format(epoch, final_loss,(t.time() - t_2)))

    print('Total Training loss {}, Time  {}'.format(final_loss, (t.time() - t_1)))

    return final_loss


if __name__ == '__main__':


    image_size = 512
    device = 'cpu'
    # Setup parser
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
    parser.add_argument('-cuda', '--cuda', default='Y')


    # Optional training parameters (when need to start between epochs)
    parser.add_argument('-starting_epoch', '--starting_epoch', type=int, help="3", default=1)
    parser.add_argument('-starting_decoder', '--starting_decoder', help="#_decoder.pth")
    parser.add_argument('-starting_pickle', '--starting_pickle', help="pickledLosses.pk1", default=None)


    torch.cuda.empty_cache()

    args = parser.parse_args()

    # Import the dataset & get num of batches
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((480, 640))])
    cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    cifar_data = DataLoader(cifar_dataset, batch_size=args.batch_size, shuffle=True)

    num_batches = int(len(cifar_dataset) / args.b)

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

    # Create models
    frontend = vanilla.encoder_decoder.frontend
    encoder = vanilla.encoder_decoder.encoder
    encoder.load_state_dict(torch.load(args.l))


    vanilla_model = vanilla.vanilla_model(encoder, frontend)
    vanilla_model.to(device)
    # Define optimizer and learning rate scheduler
    optimizer = optim.Adam(vanilla_model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True, factor=0.1,
                                                     min_lr=1e-4)
    # Train the model
    train(args.e, optimizer, vanilla_model, cifar_data, scheduler, device, args.s, args.gamma,
          args.p, pickleLosses=args.starting_pickle)

    n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device,
    decoder_save = None, plot_file = None, pickleLosses = None,
    starting_epoch = 1)
