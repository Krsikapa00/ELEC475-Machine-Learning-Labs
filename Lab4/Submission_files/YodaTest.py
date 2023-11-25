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

max_ROIs = -1


if __name__ == '__main__':

    device = 'cpu'
    parser = argparse.ArgumentParser()

    parser.add_argument('-model_file', type=str, help='model weight file')
    parser.add_argument('-cuda', type=str, help='[y/N]')
    parser.add_argument('-i', metavar='input_dir', type=str, help='input dir (./)')
    parser.add_argument('-o', metavar='output_dir', type=str, help='output dir (./)')
    parser.add_argument('-IoU', metavar='IoU_threshold', type=float, help='[0.02]')
    parser.add_argument('-d', metavar='display', type=str, help='[y/N]')
    parser.add_argument('-m', metavar='mode', type=str, help='[train/test]')
    parser.add_argument('-v', default=False, type=bool, help='[train/test]')

    opt = parser.parse_args()

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

    # Load saved yodamodel
    model_file = opt.model_file
    model = yoda.model()
    model.load_state_dict(torch.load(opt.model_file))
    model.to(device=device)
    model.eval()

    print('model loaded OK!')
    print("Using device: {}".format(device))
    all_mean_IoU = []
    out_tensor = None
    print("Starting Test")
    for item in enumerate(dataset):
        idx = item[0]
        if opt.v:
            print(" {}/{}".format(idx, len(dataset)))
        # Original Kitti img
        image = item[1][0]
        label = item[1][1]
        # Get car label indx
        idx = dataset.class_label['Car']
        # Get all of the CAR ROIS for this img Truth ones
        car_ROIs = dataset.strip_ROIs(class_ID=idx, label_list=label)
        anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)
        # Generate ROIs for each image
        ROIs, boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)
        if opt.v:
            print("Created all ROIs".format())
        # Convert ROIS to a tensor
        transformed_ROI = []
        for idx, item in enumerate(ROIs):
            temp_ROI = train_transform(item)
            transformed_ROI.append(temp_ROI)
        tensor_ROI = torch.stack(transformed_ROI)

        with torch.no_grad():
            if opt.v:
                print("Putting ROI's through model".format())
            output = model(tensor_ROI)

        # Rounding results, putting answers to 1 or 0
        output_rounded = torch.round(output)
        ROI_IoUs = []
        bound_boxes = []
        # Iterate through all outputs
        if opt.v:
            print("ROIS classified as 'Car': ".format())
        for idx, item in enumerate(output_rounded):
            # Find the ones that predicted 'Car'

            if torch.argmax(item) == 1:
                if opt.v:
                    print("Index: {},    Box: {}".format(idx, boxes[idx]))
            # Get the IoU of this box from the CarROIs
                curr_roi_max_IoU = anchors.calc_max_IoU(boxes[idx], car_ROIs)
                ROI_IoUs += [curr_roi_max_IoU]
                if curr_roi_max_IoU >= IoU_threshold:
                    bound_boxes.append(boxes[idx])
        print("Printing all IoUs for each ROI of this image")
        for idx, iou in enumerate(ROI_IoUs):
            print("ROI {}, IoU calculations: {}".format(idx, ROI_IoUs[idx]))

        curr_image_mean_IoU = 0
        if len(ROI_IoUs) != 0:
            curr_image_mean_IoU = sum(ROI_IoUs)/len(ROI_IoUs)
            # sum_IoU = 0
        all_mean_IoU.append(curr_image_mean_IoU)
        print("Mean IoU {}".format(curr_image_mean_IoU))

        if show_images:
            image2 = image.copy()

            for box in bound_boxes:
                pt1 = (box[0][1], box[0][0])
                pt2 = (box[1][1], box[1][0])
                cv2.rectangle(image2, pt1, pt2, color=(0, 255, 255))

        if show_images:
            cv2.imshow('boxes', image2)
            key = cv2.waitKey(0)
            if key == ord('x'):
                break

    global_mean_IoU = sum(all_mean_IoU)/len(dataset)
    print("Final global mean IoU is: {}".format(global_mean_IoU))
