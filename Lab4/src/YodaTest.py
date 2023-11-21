import argparse
import os
#from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import YodaModel as yoda
import train
from torch.utils.data import DataLoader, Dataset
import time as t
import KiitiROIDataset as KittiData

import cv2
from KittiDataset import KittiDataset
from KittiAnchors import Anchors

save_ROIs = True
max_ROIs = -1


def calc_mean_IoU(self, ROI, ROI_list):
    sum_IoU = 0
    for i in range(len(ROI_list)):
        sum_IoU = sum_IoU + self.calc_IoU(ROI, ROI_list[i])

    mean_IoU = sum_IoU / len(ROI_list)
    return mean_IoU

def generate_ROIs(dataset, anchors):
    i = 0
    for item in enumerate(dataset):
        idx = item[0]
        image = item[1][0]
        label = item[1][1]
        # print(i, idx, label)

        idx = dataset.class_label['Car']
        #car_ROIs = dataset.strip_ROIs(class_ID=idx, label_list=label)
        # print(car_ROIs)
        # for idx in range(len(car_ROIs)):
        # print(ROIs[idx])

        anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)
        if show_images:
            image1 = image.copy()
            for j in range(len(anchor_centers)):
                x = anchor_centers[j][1]
                y = anchor_centers[j][0]
                cv2.circle(image1, (x, y), radius=4, color=(255, 0, 255))

        ROIs, boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)
        # print('break 555: ', boxes)

        #ROI_IoUs = []
        #for idx in range(len(ROIs)):
        #    ROI_IoUs += [anchors.calc_max_IoU(boxes[idx], car_ROIs)]

        # print(ROI_IoUs)

        #Removed code

    return ROIs, boxes


if __name__ == '__main__':

    image_size = 512
    device = 'cpu'

    parser = argparse.ArgumentParser()

    #parser.add_argument('-image_dir', '--image_dir', type=str, required=True,
    #                    help='Directory path to a batch of content images')

    parser.add_argument('--model_file', type=str, help='encoder/deccoder weight file')
    parser.add_argument('--cuda', type=str, help='[y/N]')
    parser.add_argument('-type', '--type', type=int, default=0)
    parser.add_argument('-b', '--b', type=int, default=512)
    parser.add_argument('-dataset', '--dataset', type=int, default=1)

    parser.add_argument('-i', metavar='input_dir', type=str, help='input dir (./)')
    parser.add_argument('-o', metavar='output_dir', type=str, help='output dir (./)')
    parser.add_argument('-IoU', metavar='IoU_threshold', type=float, help='[0.02]')
    parser.add_argument('-d', metavar='display', type=str, help='[y/N]')

    opt = parser.parse_args()
    torch.cuda.empty_cache()

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

    use_cuda = False
    if opt.cuda != None:
        if opt.cuda == 'y' or opt.cuda == 'Y':
            use_cuda = True


    #Not sure which train_transform is correct
    train_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



    #batch_size = 48 #48 ROIs generated per image

    if opt.dataset == 0:
        #cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
        #                                             transform=train_transform)
        #cifar_data = DataLoader(cifar_dataset, batch_size=opt.b, shuffle=True)
        #cifar_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
        #                                                    transform=train_transform)
        #cifar_train_data = DataLoader(cifar_dataset, batch_size=opt.b, shuffle=True)
        #frontend = model.encoder_decoder.frontend_10

        # Import the dataset
        #train_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((480, 640))])

        #image_dataset = custData.custom_dataset(opt.image_dir, train_transform)

        #num_batches = int(len(content_data) / opt.b)

        #image_data = DataLoader(image_dataset, opt.b, shuffle=True)

        test_dir = os.path.join(os.path.abspath(opt.i))


        kitti_test_dataset = KittiData.KittiROIDataset(test_dir, training=False, transform=train_transform)
        test_data = DataLoader(kitti_test_dataset, batch_size=opt.b, shuffle=False)

        #anchors = Anchors()
        #ROIs, boxes = generate_ROIs(dataset, anchors) #return roi_dataset

        #roi_dataset = KittiData.KittiROIDataset(args.roi_dir, transform= train_transform)
        num_batches = int(len(test_data) / opt.b)

        #roi_data = DataLoader(roi_dataset, batch_size= batch_size, shuffle=True)
    else:
        cifar_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                                      transform=train_transform)
        cifar_data = DataLoader(cifar_dataset, batch_size=opt.b, shuffle=True)
        cifar_train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                      transform=train_transform)
        cifar_train_data = DataLoader(cifar_train_dataset, batch_size=opt.b, shuffle=True)


    model_file = opt.model_file

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

    model = yoda.model()
    model.load_state_dict(torch.load(opt.model_file))

    model.to(device=device)
    model.eval()

    print('model loaded OK!')
    print("Using device: {}".format(device))

    out_tensor = None

    with torch.no_grad():
        top1_count = 0
        top1_train_count = 0
        top5_count = 0
        top5_train_count = 0
        print("Going through TEST dataset")

        i = 0
        for idx, data in enumerate(test_data):
            t_3 = t.time()
            imgs, label = data[0].to(device), data[1].to(device)
            out_tensor = model(imgs)

            #idx = item[0]
            #label = item[1][1]

            car_ROIs = out_tensor.strip_ROIs(class_ID=idx, label_list=label)

            ROI_IoUs = []
            for idx in range(len(ROIs)):
                ROI_IoUs += [anchors.calc_max_IoU(boxes[idx], car_ROIs)]

            sum_IoU = 0

            for ROI_val in ROI_IoUs:
                sum_IoU = sum_IoU + ROI_val

            mean_IoU = sum_IoU / len(ROI_IoUs)
            

            for k in range(len(boxes)):
                filename = str(i) + '_' + str(k) + '.png'
                if save_ROIs == True:
                    cv2.imwrite(os.path.join(output_dir, filename), ROIs[k])
                name_class = 0
                name = 'NoCar'
                if ROI_IoUs[k] >= IoU_threshold:
                    name_class = 1
                    name = 'Car'
                labels += [[filename, name_class, name]]

            if show_images:
                cv2.imshow('image', image1)
                # key = cv2.waitKey(0)
                # if key == ord('x'):
                #     break

            if show_images:
                image2 = image1.copy()

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
            i += 1
            print(i)

            if max_ROIs > 0 and i >= max_ROIs:
                break
            #
            # print(labels)
            #
        if save_ROIs == True:
            with open(os.path.join(output_dir, label_file), 'w') as f:
                for k in range(len(labels)):
                    filename = labels[k][0]
                    name_class = str(labels[k][1])
                    name = labels[k][2]
                    f.write(filename + ' ' + name_class + ' ' + name + '\n')
            f.close()




        _, top1_predicted = torch.max(out_tensor, 1)
        _, top5_predictions = torch.topk(out_tensor, 5, dim=1)

        top1_count += torch.sum(labels == top1_predicted).item()

        # for i in range(len(labels)):
        #    if labels[i] in top5_predictions[i]:
        #        top5_count += 1
        top5_count += torch.sum(top5_predictions == labels.view(-1, 1))

        print('Image #{}/{}         Time: {}'.format(idx + 1, len(cifar_data), (t.time() - t_3)))
    '''
        print("Going through training dataset")
        for idx, data in enumerate(cifar_train_data):
            t_3 = t.time()
            imgs, labels = data[0].to(device), data[1].to(device)
            out_tensor = model(imgs)

            _, top1_predicted = torch.max(out_tensor, 1)
            _, top5_predictions = torch.topk(out_tensor, 5, dim=1)

            top1_train_count += torch.sum(labels == top1_predicted).item()

            # for i in range(len(labels)):
            #    if labels[i] in top5_predictions[i]:
            #        top5_count += 1
            top5_train_count += torch.sum(top5_predictions == labels.view(-1, 1))

            print('Image #{}/{}         Time: {}'.format(idx + 1, len(cifar_train_data), (t.time() - t_3)))
        
    top1_err = 1 - top1_count / len(cifar_dataset)
    top5_err = 1 - top5_count / len(cifar_dataset)
    top1_train_err = 1 - top1_train_count / len(cifar_train_dataset)
    top5_train_err = 1 - top5_train_count / len(cifar_train_dataset)

    print(f"Top-1 Test data Error Rate: {top1_err * 100}%")
    print(f"Top-5 Test data Error Rate: {top5_err * 100}%")

    print(f"Top-1 Train Data Error Rate: {top1_train_err * 100}%")
    print(f"Top-5 Train Data Error Rate: {top5_train_err * 100}%")'''

