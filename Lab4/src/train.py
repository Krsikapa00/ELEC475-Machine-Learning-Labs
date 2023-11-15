import argparse
import pickle
import ssl
import os
import datetime

import torch.optim as optim
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import time as t

import YodaModel as model
import KiitiROIDataset as KittiData

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device,
          encoder_save=None, plot_file=None, pickleLosses = None,
          starting_epoch=1, evaluate_epochs=False, folder='./', store_data=False, test_loader=None):

    model.train()
    total_losses = []
    total_test_losses = []

    final_loss = 0.0
    final_test_loss = 0.0
    t_1 = t.time()
    print("\n=======================================")
    print("Training started at Epoch {}     @ {}\n".format(starting_epoch, datetime.datetime.now()))

    # loading_saved pickle data
    # if pickleLosses is not None:
    #     try:
    #         with open(pickleLosses, 'rb') as file:
    #             loaded_losses = pickle.load(file)
    #             (total_losses, content_losses, style_losses) = loaded_losses
    #             print("Loaded saved losses from file successfully: \n{} \n{} \n{}"
    #                   .format(total_losses, content_losses, style_losses))
    #     except Exception as e:
    #         print(f"An error occurred while loading arrays: {str(e)}")


    for epoch in range(starting_epoch, n_epochs + 1):
        print("Starting Epoch: ", epoch)
        # Losses for current epoch
        total_loss = 0.0
        total_test_loss = 0.0
        t_2 = t.time()

        for idx, data in enumerate(train_loader):

            t_3 = t.time()
            imgs, labels = data[0].to(device=device), data[1].to(device=device)
            optimizer.zero_grad()

            output_labels = model(imgs)
            loss = loss_fn(output_labels, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print('Batch #{}/{}         Time: {}'.format(idx + 1, len(train_loader), (t.time() - t_3)))
            # t_4 = t.time()

            if idx == 2:
                break


        if encoder_save is not None:
            torch.save(model.encoder.state_dict(), encoder_save)
            print("Saved frontend model under name: {}".format(encoder_save))

        scheduler.step(total_loss)
        total_losses.append(total_loss / len(train_loader))
        final_loss = total_loss / len(train_loader)

        # Evaluate model on test dataset
        test_t1 = t.time()
        if test_loader is not None:
            model.eval()
            with torch.no_grad():
                for idx, data in enumerate(test_loader):
                    # print("Time loading data:   {}".format(t.time() - t_4))
                    t_3 = t.time()
                    imgs, labels = data[0].to(device=device), data[1].to(device=device)

                    test_output = model(imgs)
                    loss = loss_fn(test_output, labels)
                    total_test_loss += loss.item()
                    if idx == 2:
                        break
            model.train()
        test_t2 = t.time() - test_t1

        total_test_losses.append(total_test_loss / len(test_loader))
        final_test_loss = total_test_loss / len(test_loader)

        # Plot loss data each epoch
        if plot_file is not None:
            plot_save = os.path.join(os.path.abspath(folder), plot_file)
            plt.figure(2, figsize=(12, 7))
            plt.clf()
            plt.plot(total_losses, label='Train')
            plt.plot(total_test_losses, label='Test')

            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loc=1)
            plt.savefig(plot_save)

        # Store loss data in pickled file
        pickleSave = os.path.join(os.path.abspath(folder), "pickled_dump.pk1")
        if pickleLosses is not None:
            pickleSave = os.path.join(os.path.abspath(folder), pickleLosses)
        try:
            with open(pickleSave, 'wb') as file:
                pickle.dump((total_losses, epoch), file)
                print("Saved losses to '{}'".format(pickleSave))
        except Exception as e:
            print(f"An error occurred while saving arrays: {str(e)}")
        print('Epoch {}, Training loss {}, Time  {}'.format(epoch, final_loss,(t.time() - t_2)))
        print('          Test loss     {}, Time  {}'.format(epoch, final_test_loss, test_t2))
        # if evaluate_epochs:


    print('Total Training loss {}, Time  {}'.format(final_loss, (t.time() - t_1)))
    print('Time finished {}'.format(datetime.datetime.now()))
    return final_loss


if __name__ == '__main__':

    # ssl._create_default_https_context = ssl._create_unverified_context

    device = 'cpu'
    # Setup parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', '--dir', type=str, required=True,
                        help='Directory path where all the Kitti_ROIs train and test images are held, dont include train or test folder')
    # training options
    parser.add_argument('-e', '--e', type=int, default=40)
    parser.add_argument('-b', '--b', type=int, default=48, help="Batch size")
    parser.add_argument('-l', '--l', help="filename to load saved encoder; Encoder.pth", required=False)
    parser.add_argument('-s', '--s', help="filename to save encoder")
    parser.add_argument('-p', '--p', help="decoder.png", default="loss_plot.png")
    parser.add_argument('-cuda', '--cuda', default='Y')
    parser.add_argument('-type', '--type', type=int, default=0)


    parser.add_argument('-lr', '--lr', type=float, default=0.001)
    parser.add_argument('-wd', '--wd', type=float, default=0.00001)
    parser.add_argument('-minlr', '--minlr', type=float, default=0.001)
    parser.add_argument('-out', '--out', default=None, help="Output folder to put all files for training run")
    parser.add_argument('-gamma', '--gamma', type=float, default=0.9)
    parser.add_argument('-dataset', '--dataset', type=int, default=1)

    args = parser.parse_args()

    print("Beginning Training for model. Using the following parameters passed (Some default)\n")
    print("\n{}".format(args))



    # Import the dataset & get num of batches
    train_dir = os.path.join(os.path.abspath(args.dir), 'train')
    test_dir = os.path.join(os.path.abspath(args.dir), 'test')

    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((150, 150))])
    kitti_train_dataset = KittiData.KittiROIDataset(train_dir, training=True, transform=train_transform)
    kitti_test_dataset = KittiData.KittiROIDataset(test_dir, training=False, transform=train_transform)

    train_data = DataLoader(kitti_train_dataset, batch_size=args.b, shuffle=True)
    test_data = DataLoader(kitti_test_dataset, batch_size=args.b, shuffle=False)


    encoder = model.encoder_decoder.encoder
    if args.l is not None:
        encoder.load_state_dict(torch.load(args.l))


    # # Creating folder to store all files made during training
    # if args.out != None:
    #     if os.path.exists(os.path.abspath(args.out)):
    #         print("Saving all files created to folder '{}'".format(args.out))
    #     else:
    #         os.mkdir(args.out)
    #         print("Created folder {} to save all files made during training".format(args.out))
    # else:
    #     args.out = "./"

    # Set the device (GPU if available, otherwise CPU)
    if torch.cuda.is_available() and args.cuda == 'Y':
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("Using device: {}".format(device))

    loss_fn = nn.functional.cross_entropy
    model = model.model(encoder)
    model.to(device)

    # Define optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True, factor=0.1,
                                                     min_lr=args.minlr)
    # TODO: DELETE TEST BELOW
    # print("Getting first img")
    # for idx, data in enumerate(train_data):
    #     print("IDX: {}    Img: {}      Label: {}".format(idx, data[0], data[1]))
    #     break


    # Train the model
    train(args.e, optimizer, model, loss_fn, train_data, scheduler, device, args.s,
           plot_file=args.p, evaluate_epochs=False, store_data=True, test_loader=test_data)