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
import numpy as np
import cv2
from torchvision.transforms import ToPILImage

from PIL import Image, ImageDraw
import NoseModel as model
import NoseDataset as NoseData

def get_euclidean_distance(predicted, confirmed, imageW=300, imageH=300):
    print("Predicted: {}\nTruth: {}".format(predicted, confirmed))

    euclidean_dis = torch.sum((confirmed - predicted) ** 2, dim=1)
    euclidean_dis = torch.sqrt(euclidean_dis)

    image_diagonal = torch.sqrt(torch.tensor(imageW ** 2) + torch.tensor(imageH ** 2))
    normalized_euclidean = (euclidean_dis/image_diagonal) * 100
    return normalized_euclidean

def eval_acc_for_epoch(data, model, device, test=False):
    total_distance = 0
    total_imgs = 0
    truePos, trueNeg, falsePos, falseNeg = 0, 0, 0, 0
    type = "Training Data"
    if test:
        type = "Test Data"
    for idx, (imgs, labels) in enumerate(data):
        imgs = imgs.to(device)

        # print("Img size: {}".format(imgs.size()))
        labels = labels.to(device)


        with torch.no_grad():
            output = model(imgs)
            # print("Output size:   {}\n Output: {}".format(output.size(), output))

        for i in range(imgs.size(0)):
            # Convert tensors to numpy arrays
            img = imgs[i].cpu().numpy().transpose((1, 2, 0))
            print("Predicted:   {}\nConfirmed: {}".format(output[i], labels[i]))
            imageScaled = cv2.resize(img, (300,300), interpolation=cv2.INTER_AREA)

            predicted = output[i].cpu().numpy()
            confirmed = labels[i].cpu().numpy()
            confirmed_xy = (int(confirmed[0] * 300),int(confirmed[1] * 300))
            predicted_xy = (float(predicted[0] * 300),float(predicted[1] * 300))
            print("Predicted:   {}\nConfirmed: {}".format(predicted_xy, confirmed_xy))
            # cv2.circle(imageScaled, nose, 2, (0, 0, 255), 1)
            cv2.circle(imageScaled, confirmed_xy, 2, (0, 0, 255), 1)

            cv2.imshow('boxes', imageScaled)
            key = cv2.waitKey(0)
            if key == ord('x'):
                break

            # Create a drawing context to draw circles
            # draw = ImageDraw.Draw(img_pil)
            #
            # # Draw a circle at the predicted coordinates (in red)
            # draw.ellipse([(predicted[0] - 5, predicted[1] - 5),
            #               (predicted[0] + 5, predicted[1] + 5)], fill='red')
            #
            # # Draw a circle at the labeled coordinates (in green)
            # draw.ellipse([(confirmed[0] - 5, confirmed[1] - 5),
            #               (confirmed[0] + 5, confirmed[1] + 5)], fill='green')

            # Display the image with marked coordinates

        # total_distance = get_euclidean_distance(output, labels)
        # total_imgs += labels.size(0)

    #     print("{} Accuracy progress: {}/{}".format(type, idx, len(data)))
    # average_distance = total_distance/total_imgs
    average_distance = 10 * 100
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

if __name__ == '__main__':

    # Setup parser
    parser = argparse.ArgumentParser()
    # Image dataset directory
    parser.add_argument('-dir', '--dir', type=str, required=True,
                        help='Directory path where all images, use -test to include whether test data is included')
    # Basic training options
    parser.add_argument('-b', '--b', type=int, default=48, help="Batch size")
    parser.add_argument('-l', '--l', help="filename to load saved encoder; Encoder.pth", required=False)
    parser.add_argument('-p', '--p', help="decoder.png", default="loss_plot.png")
    parser.add_argument('-cuda', '--cuda', default='Y')


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


    # Set the device (GPU if available, otherwise CPU)
    if torch.cuda.is_available() and args.cuda == 'Y':
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("Using device: {}".format(device))

    local_model.to(device)
    avg_acc, avg_test_acc = evaluate_epoch_acc(local_model, train_data, device, test_loader=test_data)


# python ./src/test.py -