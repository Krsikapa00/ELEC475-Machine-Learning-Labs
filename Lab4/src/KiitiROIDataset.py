import os
import fnmatch
import torch
from torch.utils.data import Dataset
from torchvision import transforms, models
import cv2

class KittiROIDataset(Dataset):
    def __init__(self, dir, training=True, transform=None):
        self.dir = dir
        self.training = training
        self.mode = 'train'
        self.target_transform = lambda data: torch.tensor(data, dtype=torch.int)
        if self.training == False:
            self.mode = 'test'
        self.img_dir = os.path.join(dir)
        self.label_dir = os.path.join(dir)
        self.transform = transform
        self.num = 0
        self.img_files = []
        self.labels = {}
        for file in os.listdir(self.img_dir):
            if fnmatch.fnmatch(file, '*.png'):
                self.img_files += [file]

        label_path = os.path.join(self.label_dir, 'labels.txt')
        labels_string = None
        with open(label_path) as label_file:
            labels_string = label_file.readlines()
        for i in range(len(labels_string)):
            lsplit = labels_string[i].split(' ')
            self.labels[lsplit[0]] = int(lsplit[1])
        print("Filled out Labels")
        self.max = len(self)

        # print('break 12: ', self.img_dir)
        # print('break 12: ', self.label_dir)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        #TODO NICK: Update this to return label and img correctly
        '''
        Get img number from name
        open labels.txt file from directory
        iterate through each line in file
        Find line that starts w image name
        Check what value is on the line. return the integer (0, 1)
        '''
        img_name = os.path.splitext(self.img_files[idx])[0]
        # print("IMAGE NAME: {}".format(img_name))
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if self.transform is not None:
            image = self.transform(image)

        label = int(self.labels[self.img_files[idx]])

        if label is None:
            print("Could not find a label for this ROI image. Please check labels.txt file")
            label = 0

        return image, label

    def __iter__(self):
        self.num = 0
        return self

    def __next__(self):
        if (self.num >= self.max):
            raise StopIteration
        else:
            self.num += 1
            return self.__getitem__(self.num-1)


    def strip_ROIs(self, class_ID, label_list):
        ROIs = []
        for i in range(len(label_list)):
            ROI = label_list[i]
            if ROI[1] == class_ID:
                pt1 = (int(ROI[3]),int(ROI[2]))
                pt2 = (int(ROI[5]), int(ROI[4]))
                ROIs += [(pt1,pt2)]
        return ROIs
