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



def get_output_accuracy(pred, confirmed, top_res=(1,)):
    max_num_results = max(top_res) #get highest result looking for
    data_size = confirmed.size(0)

    pred_values, pred_indx = pred.topk(k=max_num_results, dim=1)
    pred_indx = pred_indx.t()
#   Reshape confirmed indxs
    confirmed_indx_reshape = confirmed.view(1, -1).expand_as(pred_indx)
    answers = pred_indx == confirmed_indx_reshape

    top_res_acc = []
    for k in top_res:
        temp_answers = answers[:k] #limit each image answer to k values
        temp_answers = temp_answers.reshape(-1).float() #flatten to see whats right form ALL images in batch
        temp_answers = temp_answers.float().sum(dim=0, keepdim=True)
        curr_acc = temp_answers/data_size
        top_res_acc.append(curr_acc)
    return top_res_acc

def eval_acc_for_epoch(data, model, device, test=False):
    tot_top1_accuracy = 0
    truePos, trueNeg, falsePos, falseNeg = 0
    type = "Training Data"
    if test:
        type = "Test Data"
    for idx, (imgs, labels) in enumerate(data):
        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output = model(imgs)
        [curr_top1] = get_output_accuracy(output, labels, (1,))
        tot_top1_accuracy += curr_top1
        output_rounded = torch.round(output)
        truePos += torch.sum((output_rounded == 1) & (labels == 1)).item()
        trueNeg += torch.sum((output_rounded == 0) & (labels == 0)).item()
        falsePos += torch.sum((output_rounded == 1) & (labels == 0)).item()
        falseNeg += torch.sum((output_rounded == 0) & (labels == 1)).item()

        # if idx == 1:
        #     break

        print("{} Accuracy progress: {}/{}".format(type, idx, len(data)))
    return tot_top1_accuracy, (truePos, trueNeg, falsePos, falseNeg)
def evaluate_epoch_acc(model, data, device, test_loader=None):
    model.eval() #Set to evaluate
    tot_top1_accuracy = 0
    test_tot_top1_accuracy = 0
    print("Accuracy for Training Data:")
    tot_top1_accuracy, confusion_matrix = eval_acc_for_epoch(data, model, device)
    avg_test_top1_epoch_acc = 0
    print("Accuracy for Test Data:")

    if test_loader is not None:
        test_tot_top1_accuracy = eval_acc_for_epoch(test_loader, model, device, True)
    avg_test_top1_epoch_acc = test_tot_top1_accuracy / len(test_loader)
    avg_top1_epoch_acc = tot_top1_accuracy / len(data)

    model.train() #Set to train when returning to function

    return avg_top1_epoch_acc, avg_test_top1_epoch_acc



def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device,
          encoder_save=None, plot_file=None, pickleLosses = None,
          starting_epoch=1, evaluate_epochs=False, folder='./', store_data=False, test_loader=None):

    model.train()
    total_losses = []
    total_test_losses = []
    total_top1_accuracy = []
    test_total_top1_accuracy = []

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
                (total_losses, total_test_losses, epoch) = loaded_losses
                print("Loaded saved losses from file successfully: \n{} \n{}"
                      .format(total_losses, total_test_losses))
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
            loss = loss_fn(output_labels, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print('Batch #{}/{}         Time: {}'.format(idx + 1, len(train_loader), (t.time() - t_3)))
            # t_4 = t.time()
            # if idx == 2:
            #     break

        if encoder_save is not None:
            save_name = os.path.join(os.path.abspath(folder), encoder_save)
            save_name_entire = os.path.join(os.path.abspath(folder), 'full_' + encoder_save)
            torch.save(model.encoder.state_dict(), save_name)
            torch.save(model.state_dict(), save_name_entire)
            print("Saved frontend model under name: {}".format(encoder_save))

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
                    # print("Time loading data:   {}".format(t.time() - t_4))
                    t_3 = t.time()
                    imgs, labels = data[0].to(device=device), data[1].to(device=device)

                    test_output = model(imgs)
                    loss = loss_fn(test_output, labels)
                    total_test_loss += loss.item()
                    # if idx == 2:
                    #     break
                    print('Test Batch #{}/{}         Time: {}'.format(idx + 1, len(test_loader), (t.time() - t_3)))

            model.train()
        test_t2 = t.time() - test_t1

        total_test_losses.append(total_test_loss / len(test_loader))
        final_test_loss = total_test_loss / len(test_loader)

        if evaluate_epochs:
            print("Testing Model Accuracy for Epoch: {}".format(epoch))
            top_1_acc, test_top_1_acc = evaluate_epoch_acc(model, train_loader, device, test_loader=test_loader)
            total_top1_accuracy.append(float(top_1_acc))
            print("Accuracy (Training Data) = TOP-1:   {}  ".format(top_1_acc))
            if test_loader is not None:
                test_total_top1_accuracy.append(float(test_top_1_acc))
            print("Accuracy (Test Data) = TOP-1:   {} ".format(test_top_1_acc))

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

                pickle.dump((total_losses, total_test_losses, total_top1_accuracy, test_total_top1_accuracy, epoch)
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


    parser.add_argument('-lr', '--lr', type=float, default=0.001)
    parser.add_argument('-wd', '--wd', type=float, default=0.00001)
    parser.add_argument('-minlr', '--minlr', type=float, default=0.001)
    parser.add_argument('-out', '--out', default=None, help="Output folder to put all files for training run")
    parser.add_argument('-gamma', '--gamma', type=float, default=0.9)
    parser.add_argument('-start_epoch', '--start_epoch', type=int, default=1)
    parser.add_argument('-start_pickle', '--start_pickle', default=None)

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

    local_model = model.model()

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True, factor=0.1,
                                                     min_lr=args.minlr)
    # TODO: DELETE TEST BELOW
    # print("Getting first img")
    # for idx, data in enumerate(train_data):
    #     print("IDX: {}    Img: {}      Label: {}".format(idx, data[0], data[1]))
    #     break


    # Train the model
    train(args.e, optimizer, local_model, loss_fn, train_data, scheduler, device, args.s, pickleLosses=args.start_pickle,
          plot_file=args.p, evaluate_epochs=True, store_data=True, test_loader=test_data, folder=args.out,
          starting_epoch=args.start_epoch)
