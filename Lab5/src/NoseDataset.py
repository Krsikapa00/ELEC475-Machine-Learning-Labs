import os
import fnmatch
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2

class NoseDataset(Dataset):
    def __init__(self, dir, training=True, transform=None):
        self.dir = dir
        self.training = training
        self.mode = 'train'
        label_files = ['train_noses.2.txt', 'train_noses.3.txt']

        if self.training == False:
            self.mode = 'test'
            label_files = ['test_noses.txt']
        self.img_dir = os.path.join(dir)
        self.label_dir = os.path.join(dir)
        self.transform = transform
        self.num = 0
        self.img_files = []
        self.labels = {}
        for file in os.listdir(self.img_dir):
            if fnmatch.fnmatch(file, '*.png'):
                self.img_files += [file]

        for file in label_files:

            label_path = os.path.join(self.label_dir, file)
            with open(label_path) as label_file:
                labels_string = label_file.readlines()
            for i in range(len(labels_string)):
                #Example Label: Bombay_6.jpg,"(244, 203)"
                #Split with , to separate image name and coordinates.
                lsplit = labels_string[i].split(',')
                # Converts string wrapped tuple into tuple?
                curr_label = eval(lsplit[1])
                self.labels[lsplit[0]] = curr_label
        self.max = len(self)


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path)
        # image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if self.transform is not None:
            image = self.transform(image)
        originalLabels = self.labels[self.img_files[idx]]
        normalizedLabels = [originalLabels[0]/image.width, originalLabels[1]/image.height]
        label = torch.tensor(normalizedLabels, dtype=torch.float32)

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