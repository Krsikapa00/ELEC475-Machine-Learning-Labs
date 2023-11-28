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
        label_files = 'train_noses.3.txt'

        if self.training == False:
            self.mode = 'test'
            label_files = 'test_noses.txt'
        self.img_dir = os.path.join(dir)
        self.label_dir = os.path.join(dir)
        self.transform = transform
        self.num = 0
        self.img_files = []
        self.labels = {}


        labels_string = []
        label_path = os.path.join(self.label_dir, label_files)
        with open(label_path, 'r') as label_file:
            labels_string = label_file.readlines()
        for i in range(len(labels_string)):

            #Example Label: Bombay_6.jpg,"(244, 203)"
            #Split with , to separate image name and coordinates.
            lsplit = labels_string[i].rstrip('\n')
            lsplit = lsplit.split(',', maxsplit=1)
            # ['Bombay_6.jpg', '"(244, 203)"']
            # Check if image exists.
            img_path = os.path.join(os.path.abspath(self.img_dir), lsplit[0])

            if os.path.exists(img_path):

                # Image and label exists. Add them both
                self.img_files += [lsplit[0]]
                # print(lsplit)
                self.labels[lsplit[0]] = lsplit[1]
        print("Length of imgs: {}".format(len(self.img_files)))
        print("Length of labels: {}".format(len(self.labels)))
        self.max = len(self)


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = ''
        try:
            img_path = os.path.join(self.img_dir, self.img_files[idx])
            # image1 = Image.open(img_path)
            img_name = img_path
            image = cv2.imread(img_path)

            img_width = image.data.shape[1]
            img_height = image.data.shape[0]

            if self.transform is not None:
                image = self.transform(image)

            original_labels = self.labels[self.img_files[idx]]
            curr_label = eval(original_labels)

            tempLabel = curr_label[1:-1]
            tempLabel = tempLabel.split(", ")
            # print("Name: {}".format(img_name))
            # print("Shape: {}    {}".format(img_width, img_height))
            #
            # print("Original Labels: {}".format(tempLabel))

            normalized_labels = [float(tempLabel[0])/img_width, float(tempLabel[1])/img_height]
            # print("Normalized Labels: {}".format(normalized_labels))

            label = torch.tensor(normalized_labels, dtype=torch.float32)

            if label is None:
                print("Could not find a label for this ROI image. Please check labels.txt file")
                label = 0
            return image, label
        except Exception as e:
            print("Error caught at __get_item__   {}:   {}".format(img_name, e))
            return None, None


    def __iter__(self):
        self.num = 0
        return self

    def __next__(self):
        if (self.num >= self.max):
            raise StopIteration
        else:
            self.num += 1
            return self.__getitem__(self.num-1)