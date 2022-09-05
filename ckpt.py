import torch
import torch.nn as nn
from torchvision import transforms
from model import VGG16_revised, VGG16_pre, LeNet5
from preprocess import CreateList, CustomDataset
from train_pipline import train

# Hyperparameters
LR = 0.001
batch_size = 32
epoch = 5

# Training images path & label file
dir_img_train = './train_images'
path_label_train = 'train.csv'

# Split image list and label list into train and valid.
train_list = CreateList(dir_img_train, path_label_train, shuffle=True)
train_valid_split = round(train_list.length * 0.8)
train_img = train_list.img[:train_valid_split]
train_label = train_list.label[:train_valid_split]
valid_img = train_list.img[train_valid_split:]
valid_label = train_list.label[train_valid_split:]

# Image preprocessing
transform = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
}

# Create DataLoader
train_dataset = CustomDataset(train_img, train_label, transform['train'])
valid_dataset = CustomDataset(valid_img, valid_label, transform['valid'])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=0,
                                           pin_memory=True)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=0,
                                           pin_memory=True)

'''
==============================================================================
======================= Training from the check-point ========================
==============================================================================
'''
' Using the check point to keep training'
# net = VGG16_pre()
# net = LeNet5()
net = VGG16_revised()

# optimize all parameters in cnn
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

# loading check-point information
checkpoint = torch.load("check_phase_vgg.pt")
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
loss_fnc = nn.CrossEntropyLoss()  # target label isn't one-hotted

# keep training
train_acc_his, valid_acc_his, train_losses_his, valid_losses_his, model1 = \
 train(net, epoch, train_loader, valid_loader, optimizer, loss_fnc, batch_size,  "check_phase_vgg.pt")