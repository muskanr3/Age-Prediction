import numpy as np
import pandas as pd
from glob import glob
from os.path import join
from torchvision.transforms import RandomApply
from torchvision.transforms import RandomRotation, RandomHorizontalFlip, ColorJitter, GaussianBlur
# from torchvision.transforms import RandomRotation, RandomHorizontalFlip, ColorJitter, GaussianBlur

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.optim as optim

class AgeDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, annot_path, train=True):
        super(AgeDataset, self).__init__()

        self.annot_path = annot_path
        self.data_path = data_path
        self.train = train

        self.ann = pd.read_csv(annot_path)
        self.files = self.ann['file_id']
        if train:
            self.ages = self.ann['age']
            self.min_age = min(self.ages)
            self.max_age = max(self.ages)
            # Scale ages
            self.ages = self.scale_ages(self.ages)
        self.transform = self._transform(224)

    @staticmethod    
    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    def _transform(self, n_px):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return Compose([
            Resize(n_px),
            self._convert_image_to_rgb,
            RandomApply([RandomRotation(10)], p=0.5),
            RandomApply([GaussianBlur(kernel_size=3)], p=0.5),
            ToTensor(),
            Normalize(mean, std),
        ])

    def scale_ages(self, ages):
        return (ages - self.min_age) / (self.max_age - self.min_age)

    def descale_ages(self, scaled_ages):
        return scaled_ages * (self.max_age - self.min_age) + self.min_age

    def read_img(self, file_name):
        im_path = join(self.data_path,file_name)   
        img = Image.open(im_path)
        img = self.transform(img)
        return img

    def __getitem__(self, index):
        file_name = self.files[index]
        img = self.read_img(file_name)
        if self.train:
            age = self.ages[index]
            return img, age
        else:
            return img

    def __len__(self):
        return len(self.files)

train_path = '../faces_dataset/train'
train_ann = '../faces_dataset/train.csv'
train_dataset = AgeDataset(train_path, train_ann, train=True)


test_path = '../faces_dataset/test'
test_ann = '../faces_dataset/submission.csv'
test_dataset = AgeDataset(test_path, test_ann, train=False)


train_indices, valid_indices = train_test_split(range(len(train_dataset)), test_size=0.1, random_state=42)

train_subset = Subset(train_dataset, train_indices)
valid_subset = Subset(train_dataset, valid_indices)

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(valid_subset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

from torch.optim import lr_scheduler

all_preds = []

for i in range(2):
    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=1)

    # Replace classifier
    num_ftrs = model._fc.in_features
    model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256), 
            nn.ReLU(),                      # ReLU activation function
            nn.Linear(256, 1)               # Final fully connected layer with a single output neuron for age prediction
        )

    criterion = nn.L1Loss()  #mae loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    epochs = 10
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.unsqueeze(1)

            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.unsqueeze(1)

                outputs = model(inputs)

                # Descale the predicted ages to calc loss
                outputs = outputs.squeeze(1)
                outputs_descaled = train_dataset.descale_ages(outputs.cpu().numpy())
                targets_descaled = train_dataset.descale_ages(targets.cpu().numpy())

                loss = criterion(torch.tensor(outputs_descaled).float(), torch.tensor(targets_descaled).squeeze().float())
                val_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        

        scheduler.step()
        
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        
    @torch.no_grad()
    def predict(loader, model, dataset):
        model.eval()
        predictions = []

        for img in tqdm(loader):
            img = img.to(device)

            pred = model(img)
            pred_descaled = dataset.descale_ages(pred.cpu().numpy().flatten())
            predictions.extend(pred_descaled)

        return predictions

    preds = predict(test_loader, model, train_dataset)

    all_preds.append(preds)
    
############## diff model starts here ########

model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=1)


num_ftrs = model._fc.in_features
model.fc = nn.Linear(num_ftrs, 1)


criterion = nn.L1Loss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)

from torch.optim import lr_scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


epochs = 10
model.to(device)
print("model 3/3 starting")

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    
    for inputs, targets in tqdm(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.unsqueeze(1)

        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.float())
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.unsqueeze(1)

            outputs = model(inputs)

            # Descale the predicted ages
            outputs = outputs.squeeze(1)
            outputs_descaled = train_dataset.descale_ages(outputs.cpu().numpy())
            targets_descaled = train_dataset.descale_ages(targets.cpu().numpy())

            loss = criterion(torch.tensor(outputs_descaled).float(), torch.tensor(targets_descaled).squeeze().float())
            val_loss += loss.item() * inputs.size(0)
    
    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    

    scheduler.step()
    

    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

       
@torch.no_grad()
def predict(loader, model, dataset):
    model.eval()
    predictions = []

    for img in tqdm(loader):
        img = img.to(device)

        pred = model(img)
        pred_descaled = dataset.descale_ages(pred.cpu().numpy().flatten())
        predictions.extend(pred_descaled)

    return predictions

preds3 = predict(test_loader, model, train_dataset)
all_preds.append(preds3)

# taking mean of all pred ages from the 3 models 
preds = np.mean(all_preds, axis=0)


submit = pd.read_csv('../faces_dataset/submission.csv')
submit['age'] = preds
submit.to_csv('baseline-ensemble-new-mean.csv', index=False)
