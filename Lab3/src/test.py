import argparse
import os
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import vanilla_model as vanilla
import moded_model as moded
import train
from torch.utils.data import DataLoader, Dataset
import time as t


def input_num():
    num = int(input('Input a number between 0 and 10: '))
    while True:
        if num > 10 or num < 0:
            num = int(input('Out of Bounds. Input a number between 0 and 10: '))
        else:
            return num


def top1_error(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    top1_err = 1 - correct / len(labels)
    return top1_err


def top5_error(predictions, labels):
    _, top5_predictions = torch.topk(predictions, 5, 1, largest=True, sorted=True)
    correct = 0
    for i in range(len(labels)):
        if labels[i] in top5_predictions[i]:
            correct += 1
    top5_err = 1 - correct / len(labels)
    return top5_err


if __name__ == '__main__':

    image_size = 512
    device = 'cpu'

    parser = argparse.ArgumentParser()

    parser.add_argument('--encoder_file', type=str, help='encoder weight file')
    parser.add_argument('--decoder_file', type=str, help='decoder weight file')
    parser.add_argument('--cuda', type=str, help='[y/N]')
    parser.add_argument('-type', '--type', type=int, default=0)
    parser.add_argument('-b', '--b', type=int, default=512)
    parser.add_argument('-dataset', '--dataset', type=int, default=1)

    opt = parser.parse_args()

    if opt.type == 0:
        model_type = vanilla
    else:
        model_type = moded

    train_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if opt.dataset == 0:
        cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                     transform=train_transform)
        cifar_data = DataLoader(cifar_dataset, batch_size=opt.b, shuffle=True)
        frontend = model_type.encoder_decoder.frontend_10

    else:
        cifar_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                                      transform=train_transform)
        cifar_data = DataLoader(cifar_dataset, batch_size=opt.b, shuffle=True)
        frontend = model_type.encoder_decoder.frontend

    decoder_file = opt.decoder_file
    encoder_file = opt.encoder_file

    use_cuda = False
    if opt.cuda == 'y' or opt.cuda == 'Y':
        use_cuda = True


    if torch.cuda.is_available() and opt.cuda == 'Y':
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("Using device: {}".format(device))

    encoder = model_type.encoder_decoder.encoder
    encoder.load_state_dict(torch.load(encoder_file))

    frontend.load_state_dict(torch.load(decoder_file))
    model = model_type.model(encoder, frontend)

    total_losses = []
    total_top1_accuracy = []
    total_top5_accuracy = []
    final_loss = 0.0

    model.to(device=device)
    model.eval()

    print('model loaded OK!')
    print("Using device: {}".format(device))

    out_tensor = None
    with torch.no_grad():
        top1_count = 0
        top5_count = 0

        for idx, data in enumerate(cifar_data):
            t_3 = t.time()
            imgs, labels = data[0].to(device), data[1].to(device)
            out_tensor = model(imgs)

            _, top1_predicted = torch.max(out_tensor, 1)
            _, top5_predictions = torch.topk(out_tensor, 5, 1, largest=True, sorted=True)

            # if labels[idx] == top1_predicted:
            #	top1_count = top1_count + 1

            # if label[idx] in top5_predictions:
            #	top5_count = top5_count + 1

            top1_count += torch.sum(labels == top1_predicted).item()

            for i in range(len(labels)):
                if labels[i] in top5_predictions[i]:
                    top5_count += 1

            print('Image #{}/{}         Time: {}'.format(idx + 1, len(cifar_data), (t.time() - t_3)))

    top1_err = 1 - top1_count / len(cifar_dataset)
    top5_err = 1 - top5_count / len(cifar_dataset)

    print(f"Top-1 Error Rate: {top1_err * 100}%")
    print(f"Top-5 Error Rate: {top5_err * 100}%")
