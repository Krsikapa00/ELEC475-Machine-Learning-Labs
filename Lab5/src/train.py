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

import NoseModel as model
import NoseDataset as NoseData

def get_euclidean_distance(predicted, confirmed, imageW=300, imageH=300):

    euclidean_dis = torch.sum((confirmed - predicted) ** 2, dim=1)
    euclidean_dis = torch.sqrt(euclidean_dis)

    image_diagonal = torch.sqrt(torch.tensor(imageW ** 2) + torch.tensor(imageH ** 2))
    normalized_euclidean = (euclidean_dis/image_diagonal) * 100
    return normalized_euclidean

def euclidean_loss_fn(predicted, confirmed):
    # print("Predicted: {}".format(predicted))
    # print("Confirmed: {}".format(confirmed))
    # print("Size: {}".format(confirmed.size()))
    # x_diff = (predicted - confirmed) ** 2
    euclidean_dis = torch.sum((confirmed - predicted) ** 2, dim=1)
    euclidean_dis = torch.sqrt(euclidean_dis).sum()
    euclidean_dis = euclidean_dis/confirmed.size(0)
    # print("Diff, squared: {}".format(euclidean_dis))
    return euclidean_dis

def eval_acc_for_epoch(data, model, device, test=False):
    total_distance = 0
    total_imgs = 0
    truePos, trueNeg, falsePos, falseNeg = 0, 0, 0, 0
    type = "Training Data"
    if test:
        type = "Test Data"
    for idx, (imgs, labels) in enumerate(data):
        imgs = imgs.to(device)
        print("Img size: {}".format(imgs.size(0)))
        labels = labels.to(device)

        with torch.no_grad():
            output = model(imgs)
        total_distance = get_euclidean_distance(output, labels)
        total_imgs += labels.size(0)

        print("{} Accuracy progress: {}/{}".format(type, idx, len(data)))
    average_distance = total_distance/total_imgs
    average_distance = average_distance * 100
    return average_distance
def evaluate_epoch_acc(model, data, device, test_loader=None):
    model.eval() #Set to evaluate
    test_accuracy = 0
    train_accuracy = 0
    print("Accuracy for Training Data:")
    train_accuracy = eval_acc_for_epoch(data, model, device)
    if test_loader is not None:
        print("Accuracy for Test Data:")
        test_accuracy = eval_acc_for_epoch(test_loader, model, device, True)

    model.train() #Set to train when returning to function
    print("Accuracies:   {}\nTest:   {}".format(train_accuracy, test_accuracy))
    return train_accuracy, test_accuracy



def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device,
          encoder_save=None, plot_file=None, pickleLosses = None,
          starting_epoch=1, evaluate_epochs=False, folder='./', store_data=False, test_loader=None):

    model.train()
    total_losses = []
    total_test_losses = []
    train_accuracy = []
    test_accuracy = []

    final_loss = 0.0
    final_test_loss = 0.0
    t_1 = t.time()
    print("\n=======================================")
    print("Training started at Epoch {}     @ {}\n".format(starting_epoch, datetime.datetime.now()))

    # loading_saved pickle data
    if pickleLosses is not None:
        try:
            with open(pickleLosses, 'rb') as file:
                loaded_losses = pickle.load(file)
                (total_losses, total_test_losses, train_accuracy, test_accuracy, epoch) = loaded_losses
                print("Loaded saved losses from file successfully: \n{} \n{}, \n{}, \n{}"
                      .format(total_losses, total_test_losses, train_accuracy, test_accuracy))
        except Exception as e:
            print(f"An error occurred while loading arrays: {str(e)}")


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
            loss = euclidean_loss_fn(output_labels, labels)
            # loss2 = loss_fn(output_labels, labels)
            # print("Losses: {}    {}".format(loss, loss2))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print('Batch #{}/{}         Time: {}'.format(idx + 1, len(train_loader), (t.time() - t_3)))

        if encoder_save is not None and (epoch % 20 == 0 or epoch == n_epochs):
            save_name = os.path.join(os.path.abspath(folder), encoder_save)
            torch.save(model.state_dict(), save_name)
            print("Saved model under name: {}".format(encoder_save))

        scheduler.step(total_loss)
        total_losses.append(total_loss / len(train_loader))
        final_loss = total_loss / len(train_loader)

        # Evaluate model on test dataset
        test_t1 = t.time()
        if test_loader is not None:
            print("Evaluating model against Test Data for epoch: {}".format(epoch))
            model.eval()
            with torch.no_grad():
                for idx, data in enumerate(test_loader):
                    t_3 = t.time()
                    imgs, labels = data[0].to(device=device), data[1].to(device=device)

                    test_output = model(imgs)
                    loss = euclidean_loss_fn(test_output, labels)
                    # loss = loss_fn(test_output, labels)
                    total_test_loss += loss.item()
                    print('Validation Batch #{}/{}         Time: {}'.format(idx + 1, len(test_loader), (t.time() - t_3)))

            model.train()
        test_t2 = t.time() - test_t1

        total_test_losses.append(total_test_loss / len(test_loader))
        final_test_loss = total_test_loss / len(test_loader)

        if evaluate_epochs:
            print("Getting Model Accuracy for Epoch: {}".format(epoch))
            curr_epoch_accuracy, test_curr_epoch_accuracy = (
                evaluate_epoch_acc(model, train_loader, device, test_loader=test_loader))

            train_accuracy.append(float(curr_epoch_accuracy))
            print("Accuracy (Training Data) = TOP-1:   {}  ".format(curr_epoch_accuracy))
            if test_loader is not None:
                test_accuracy.append(float(test_curr_epoch_accuracy))
            print("Accuracy (Test Data) = TOP-1:   {} ".format(test_curr_epoch_accuracy))

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
        encdoer_prefix = encoder_save[:-4]
        pickleSave = os.path.join(os.path.abspath(folder), (str(encdoer_prefix) + "_pickled.pk1"))
        if pickleLosses is not None:
            pickleSave = os.path.join(os.path.abspath(folder), pickleLosses)
        try:
            with open(pickleSave, 'wb') as file:

                pickle.dump((total_losses, total_test_losses, train_accuracy, test_accuracy, epoch)
                            , file)

                print("Saved losses to '{}'".format(pickleSave))
        except Exception as e:
            print(f"An error occurred while saving arrays: {str(e)}")
        print('Epoch {}, Training loss {}, Time  {}'.format(epoch, final_loss,(t.time() - t_2)))
        print('          Test loss     {}, Time  {}'.format(final_test_loss, test_t2))

    print('Total Training loss {}      Test Training Loss:    {}    ,  Time  {}'
          .format(final_loss, final_test_loss, (t.time() - t_1)))
    print('Time finished {}'.format(datetime.datetime.now()))
    return final_loss, final_test_loss


if __name__ == '__main__':

    device = 'cpu'
    # Setup parser
    parser = argparse.ArgumentParser()
    # Image dataset directory
    parser.add_argument('-dir', '--dir', type=str, required=True,
                        help='Directory path where all images, use -test to include whether test data is included')
    # Basic training options
    parser.add_argument('-e', '--e', type=int, default=40, help="Num of Epochs")
    parser.add_argument('-b', '--b', type=int, default=48, help="Batch size")
    parser.add_argument('-l', '--l', help="filename to load saved encoder; Encoder.pth", required=False)
    parser.add_argument('-s', '--s', help="filename to save encoder", default="encoder.pth")
    parser.add_argument('-p', '--p', help="decoder.png", default="loss_plot.png")
    parser.add_argument('-cuda', '--cuda', default='Y')

    # Hyper parameters options
    parser.add_argument('-lr', '--lr', type=float, default=0.001)
    parser.add_argument('-wd', '--wd', type=float, default=0.00001)
    parser.add_argument('-minlr', '--minlr', type=float, default=0.001)
    parser.add_argument('-gamma', '--gamma', type=float, default=0.9)

    #Options to continue/save files
    parser.add_argument('-out', '--out', default=None, help="Output folder to put all files for training run")
    parser.add_argument('-start_epoch', '--start_epoch', type=int, default=1)
    parser.add_argument('-start_pickle', '--start_pickle', default=None)
    args = parser.parse_args()

    print("Beginning Training for model. Using the following parameters passed (Some default)\n")
    print("\n{}".format(args))

    # Import the dataset & get num of batches
    image_dir = os.path.abspath(args.dir)

    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((300, 300))])
    train_dataset = NoseData.NoseDataset(image_dir, training=True, transform=train_transform)
    test_dataset = NoseData.NoseDataset(image_dir, training=False, transform=train_transform)

    train_data = DataLoader(train_dataset, batch_size=args.b, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=args.b, shuffle=False)
    encoder = model.encoder_decoder.encoder
    local_model = model.model(encoder)

    if args.l is not None:
        local_model.load_state_dict(torch.load(args.l))

    # # Creating folder to store all files made during training
    if args.out != None:
        if os.path.exists(os.path.abspath(args.out)):
            print("Saving all files created to folder '{}'".format(args.out))
        else:
            os.mkdir(args.out)
            print("Created folder {} to save all files made during training".format(args.out))
    else:
        args.out = "./"

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
    local_model.to(device)

    # Define optimizer and learning rate scheduler
    optimizer = optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=args.wd)
    # optimizer = optim.SGD(local_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True, factor=0.01,
                                                     min_lr=args.minlr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=args.gamma)
    # Train the model
    train(args.e, optimizer, local_model, loss_fn, train_data, scheduler, device, args.s, pickleLosses=args.start_pickle
          , plot_file=args.p, evaluate_epochs=False, test_loader=test_data, folder=args.out,
          starting_epoch=args.start_epoch)
