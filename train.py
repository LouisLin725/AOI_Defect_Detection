import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import time
from preprocess import CreateList, CustomDataset
from model import VGG16_revised, VGG16_pre, LeNet5
from train_pipline import train

# Hyperparameters
n_epochs = 1  # Epochs
batch_size = 32  # Batch：number of images per batch
LR = 0.001  # Learning Rate

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
=============================== Main training ================================
==============================================================================
'''
model1 = LeNet5()
# model1 = VGG16_pre()
# model1 = VGG16_revised()
print(model1)  # model structure

# optimize all parameters in cnn
optimizer = torch.optim.Adam(model1.parameters(), lr=LR)
loss_fnc = nn.CrossEntropyLoss()  # target label isn't one-hotted

# Training the model and calculating training time
start = time.time()

train_acc_his, valid_acc_his, train_losses_his, valid_losses_his, model1 = \
    train(model1, n_epochs, train_loader, valid_loader, optimizer,
          loss_fnc, batch_size, "check_phase_lenet.pt")

end = time.time()
print('Training time = ' + str(end - start))

torch.save(model1, 'lenet_best_model.pth')  # Save the model

# Plot accuracy
plt.figure()
plt.plot(train_acc_his,linewidth=2.5)
plt.plot(valid_acc_his,linewidth=2.5)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=20,
           loc='lower right')
plt.xlabel('epoch', fontweight='bold', fontsize=20)
plt.ylabel('Accuracy', fontweight='bold', fontsize=20)

# Plot loss
plt.figure()
plt.plot(train_losses_his,linewidth=2.5)
plt.plot(valid_losses_his,linewidth=2.5)
plt.legend(['Training loss', 'Validation loss'], fontsize=20,
           loc='upper right')
plt.xlabel('epoch', fontweight='bold', fontsize=20)
plt.ylabel('Loss', fontweight='bold', fontsize=20)

