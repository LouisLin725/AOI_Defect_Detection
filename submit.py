import torch
from torchvision import transforms
from tqdm import tqdm
import math
import pandas as pd

from preprocess import CreateList, CustomDataset

# Testing data
# Hyperparameters
batch_size = 32

# Path of testing images, test labels, and trained models
img_test_path = './test_images'
label_test_file = 'test.csv'
model_path = 'vgg_best_model.pth'

# Read the trained-model
model1 = torch.load(model_path)

# Preprocessing of the testing data
test_list = CreateList(img_test_path, label_test_file, shuffle=False,
                       train=False)

# Transform: so as training and validation data
transform = {'test': transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])}

# for the sake of fitting the arguments in CustomDataset
fake_label = [i for i in range(len(test_list.img))]

test_dataset = CustomDataset(test_list.img, label_list=fake_label,
                             transform=transform['test'])

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=0,
                                          pin_memory=True)

'''
==============================================================================
========================== Testing data prediction ===========================
==============================================================================
'''
# Make the prediction of the testing data
test_prediction = []
test_total = 0

model1.eval()
with torch.no_grad():
    for images, _ in tqdm(test_loader):
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model1(images)
        # make prediciton
        pred = output.data.max(dim=1, keepdim=True)[1]
        test_total += images.size(0)  # accumulate current data numbers
        test_prediction.append(pred)


'''
==============================================================================
========================= Construct submission file ==========================
==============================================================================
'''
# Turning the tensor-batched prediction into integer list
pred_list = []  # empty list to store int() prediction

for i in range(math.ceil(test_total / batch_size)):
    if i < math.floor(test_total / batch_size):
        for k in range(batch_size):
            pred_list.append(int(test_prediction[i][k]))
    else:
        for k in range(test_total - batch_size * i):
            pred_list.append(int(test_prediction[i][k]))

# Create the submission csv.file
dic_test = pd.DataFrame({'ID': test_list.filename, 'Label': pred_list})
upload = pd.DataFrame(dic_test)
upload.to_csv('test.csv', header=True, sep=',', encoding='utf-8', index=False)
