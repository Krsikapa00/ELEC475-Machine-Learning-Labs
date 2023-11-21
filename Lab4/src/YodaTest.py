import argparse
import os
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import YodaModel as yoda
import train
from torch.utils.data import DataLoader, Dataset
import time as t
# import custom_dataset as custData

import cv2
from KittiDataset import KittiDataset
from KittiAnchors import Anchors

save_ROIs = True
max_ROIs = -1

class CustomImageDataset(Dataset):
    def __init__(self, image_list, bounding, transform=None):
        self.image_list = image_list
        self.transform = transform
        self.bounding = bounding

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        box = self.bounding[idx]
        if self.transform:
            image = self.transform(image)

        return image, box


def calc_mean_IoU(self, ROI, ROI_list):
    sum_IoU = 0
    for i in range(len(ROI_list)):
        sum_IoU = sum_IoU + self.calc_IoU(ROI, ROI_list[i])

    mean_IoU = sum_IoU / len(ROI_list)
    return mean_IoU


def generate_ROIs(dataset, anchors):
    for item in enumerate(dataset):
        idx = item[0]
        image = item[1][0]

        anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)
        ROIs, boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)

    return ROIs, boxes


if __name__ == '__main__':

    image_size = 512
    device = 'cpu'

    parser = argparse.ArgumentParser()


    parser.add_argument('-model_file', type=str, help='model weight file')
    parser.add_argument('-cuda', type=str, help='[y/N]')
    parser.add_argument('-type', '--type', type=int, default=0)
    # parser.add_argument('-b', '--b', type=int, default=512)
    parser.add_argument('-dataset', '--dataset', type=int, default=1)

    parser.add_argument('-i', metavar='input_dir', type=str, help='input dir (./)')
    parser.add_argument('-o', metavar='output_dir', type=str, help='output dir (./)')
    parser.add_argument('-IoU', metavar='IoU_threshold', type=float, help='[0.02]')
    parser.add_argument('-d', metavar='display', type=str, help='[y/N]')
    parser.add_argument('-m', metavar='mode', type=str, help='[train/test]')

    opt = parser.parse_args()
    # torch.cuda.empty_cache()

    if torch.cuda.is_available() and opt.cuda == 'Y':
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Using device: {}".format(device))

    input_dir = None
    if opt.i != None:
        input_dir = opt.i

    output_dir = None
    if opt.o != None:
        output_dir = opt.o

    IoU_threshold = 0.02
    if opt.IoU != None:
        IoU_threshold = float(opt.IoU)

    show_images = False
    if opt.d != None:
        if opt.d == 'y' or opt.d == 'Y':
            show_images = True

    training = True
    if opt.m == 'test':
        training = False

    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((150, 150))])

    batch_size = 48  # 48 ROIs generated per image

    # Load kitti data
    dataset = KittiDataset(input_dir, training=training)
    anchors = Anchors()

    # ROIs, boxes = generate_ROIs(dataset, anchors)  # return roi_dataset
    # roi_data = DataLoader(ROIs, batch_size=batch_size, shuffle=False)

    # roi_dataset = custData.custom_dataset(ROIs, transform= train_transform)
    # num_batches = int(len(roi_dataset) / batch_size)

    # Load saved yodamodel
    model_file = opt.model_file
    model = yoda.model()
    model.load_state_dict(torch.load(opt.model_file))
    model.to(device=device)
    model.eval()

    print('model loaded OK!')
    print("Using device: {}".format(device))

    out_tensor = None
    print("Starting Test")
    for item in enumerate(dataset):
        idx = item[0]
        print("Img testing {}/{}".format(idx, len(dataset)))
        # Original Kitti img
        image = item[1][0]
        label = item[1][1]
        # Get car label indx
        idx = dataset.class_label['Car']

        # Get all of the CAR ROIS for this img Truth ones
        car_ROIs = dataset.strip_ROIs(class_ID=idx, label_list=label)

        anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)
        # Generate ROIs for each image

        print("Generating ROI's for img")
        ROIs, boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)
        print("Done getting ROI's")

        ROI_IoUs = []
        for idx in range(len(ROIs)):
            ROI_IoUs += [anchors.calc_max_IoU(boxes[idx], car_ROIs)]

        # Put ROIs through model
        roi_dataset = CustomImageDataset(ROIs, boxes, train_transform)

        with torch.no_grad():
            print("About to put ROI's through model")
            roi_data = DataLoader(roi_dataset, batch_size=48, shuffle=False)
            for data in roi_data:
                roi_imgs, labels = data[0].to(device=device), data[1]
                output = model(roi_imgs)
                print("output of model ROI imgs: {}".format(output))

                output_rounded = torch.round(output)
                print("Output rounded: {}".format(output_rounded))
                # Car_labels = (output_rounded.item() == 1)

                for idx, item in enumerate(output_rounded):
                    print("Predicted ROI {}:    {}".format(idx, torch.argmax(item)))

        sum_IoU = 0

        for ROI_val in ROI_IoUs:
            sum_IoU = sum_IoU + ROI_val
        mean_IoU = sum_IoU / len(ROI_IoUs)

        if show_images:
            image2 = image.copy()

            for k in range(len(boxes)):
                if ROI_IoUs[k] > IoU_threshold:
                    box = boxes[k]
                    pt1 = (box[0][1], box[0][0])
                    pt2 = (box[1][1], box[1][0])
                    cv2.rectangle(image2, pt1, pt2, color=(0, 255, 255))

        if show_images:
            cv2.imshow('boxes', image2)
            key = cv2.waitKey(0)
            if key == ord('x'):
                break



