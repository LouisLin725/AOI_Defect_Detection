import random
import torch
from PIL import Image


#  Combine images with their corresponding labels
class CreateList():

    def __init__(self, dir_img, path_label=None, header=True, shuffle=False, train=True):
        self.dir_img = dir_img
        self.path_label = path_label
        self.create_list(header, shuffle, train)

    def create_list(self, header, shuffle, train):

        with open(self.path_label, 'r') as f:
            if header:
                head = f.readline()  # skip header(first line)
            img_pairs = f.readlines()

        if shuffle:
            random.shuffle(img_pairs)

        self.img = []
        self.label = []
        self.filename = []

        for img_pair in img_pairs:
            img_pair = img_pair.split(',')
            self.filename.append(img_pair[0])
            self.img.append(self.dir_img + '/' + img_pair[0])
            if train:
                self.label.append(int(img_pair[1][0]))

        # Count number of imgs
        self.length = len(self.img)


# Define a custom torch.Dataset
class CustomDataset(torch.utils.data.Dataset):
    """Define a custom torch.Dataset."""

    def __init__(self, img_list, label_list=None,transform=None):

        self.data = img_list
        self.label = label_list
        self.transform = transform

    def __getitem__(self, index):
        # load one sample in a time
        image = self.data[index]
        if self.label != None:
            target = self.label[index]
        else:
            target = None
        image = Image.open(image)
        # turn image to RGB format if it is a grayscale
        if image.getbands()[0] == 'L':
            image = image.convert('RGB')
        # preprocessing data
        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        """Compute number of data."""
        return len(self.data)