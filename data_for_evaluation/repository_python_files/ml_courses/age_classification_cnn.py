import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import shutil
import os
from google.colab import drive


def download_datasets(dataset_path, training_transform, test_transform, extract_young):
    training_dataset = torchvision.datasets.CelebA(dataset_path, split='train', target_type='attr', download=False, transform=training_transform, target_transform=extract_young)
    validation_dataset = torchvision.datasets.CelebA(dataset_path, split='valid', target_type='attr', download=False, transform=test_transform, target_transform=extract_young)
    testing_dataset = torchvision.datasets.CelebA(dataset_path, split='test', target_type='attr', download=False, transform=test_transform, target_transform=extract_young)
    return training_dataset, validation_dataset, testing_dataset
    
def print_datasets_length(training_dataset, validation_dataset, testing_dataset):
    print('Training set length:', len(training_dataset))
    print('Validation set length:', len(validation_dataset))
    print('Testing set length:', len(testing_dataset))

def create_splitted_subsets(training_dataset, validation_dataset, testing_dataset):
    training_dataset = Subset(training_dataset, torch.arange(21000))
    validation_dataset = Subset(validation_dataset, torch.arange(7000))
    testing_dataset  = Subset(testing_dataset , torch.arange(7000))
    return training_dataset, validation_dataset, testing_dataset

def print_splitted_datasets(training_dataset, validation_dataset, testing_dataset):
    print('Training set:', len(training_dataset))
    print('Validation set:', len(validation_dataset))
    print('Testing set:', len(testing_dataset ))

def get_data_loaders(training_dataset, validation_dataset, testing_dataset, batch_size):
    training_data_loader = DataLoader(training_dataset, batch_size, shuffle=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size, shuffle=False)
    testing_data_loader = DataLoader(testing_dataset, batch_size, shuffle=False)
    return training_data_loader, validation_data_loader, testing_data_loader


def create_model():
    model = nn.Sequential()

    model.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1))
    model.add_module('batchnorm1', nn.BatchNorm2d(32))
    model.add_module('relu1', nn.ReLU())
    model.add_module('pool1', nn.MaxPool2d(kernel_size=2))
    model.add_module('dropout1', nn.Dropout(p=0.6))

    model.add_module('conv2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1))
    model.add_module('batchnorm2', nn.BatchNorm2d(64))
    model.add_module('relu2', nn.ReLU())
    model.add_module('pool2', nn.MaxPool2d(kernel_size=2))
    model.add_module('dropout2', nn.Dropout(p=0.4))

    model.add_module('conv3', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=1))
    model.add_module('batchnorm3', nn.BatchNorm2d(128))
    model.add_module('relu3', nn.ReLU())
    model.add_module('pool3', nn.MaxPool2d(kernel_size=2))

    model.add_module('conv4', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=1))
    model.add_module('batchnorm4', nn.BatchNorm2d(256))
    model.add_module('relu4', nn.ReLU())

    model.add_module('pool4', nn.AvgPool2d(kernel_size=4, padding=0))
    model.add_module('flatten', nn.Flatten())

    model.add_module('fc', nn.Linear(256, 1))
    model.add_module('sigmoid', nn.Sigmoid())
    
    return model


def train(model, training_data_loader, epoch, device, loss_fn, optimizer):
    training_loss_hist, training_accuracy_hist = [], []
    training_loss, training_accuracy = 0, 0

    model.train()
    for x_batch, y_batch in training_data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        prediction = model(x_batch)[:, 0]
        loss = loss_fn(prediction, y_batch.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        training_loss += loss.item()*y_batch.size(0)
        is_prediction_correct = ((prediction>=0.5).float() == y_batch).float()
        training_accuracy += is_prediction_correct.sum().cpu()

    training_loss /= len(training_data_loader.dataset)
    training_accuracy /= len(training_data_loader.dataset)
    training_loss_hist.append(training_loss)
    training_accuracy_hist.append(training_accuracy)
    print(f'Epoch {epoch+1} train accuracy: {training_accuracy:.4f}')

def eval(model, validation_data_loader, epoch, device, loss_fn, optimizer):
    validation_loss_hist, validation_accuracy_hist = [], []
    validation_loss, validation_accuracy = 0, 0
    
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in validation_data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            prediction = model(x_batch)[:, 0]
            loss = loss_fn(prediction, y_batch.float())
            validation_loss += loss.item()*y_batch.size(0)
            is_prediction_correct = ((prediction>=0.5).float() == y_batch).float()
            validation_accuracy += is_prediction_correct.sum().cpu()

    validation_loss /= len(validation_data_loader.dataset)
    validation_accuracy /= len(validation_data_loader.dataset)
    validation_loss_hist.append(validation_loss)
    validation_accuracy_hist.append(validation_accuracy)

    print(f'Epoch {epoch+1} validation accuracy: {validation_accuracy:.4f}')


def test(model, testing_data_loader, device, loss_fn, optimizer):
  testing_accuracy = 0

  model.eval()
  with torch.no_grad():
      for x_batch, y_batch in testing_data_loader:
          x_batch = x_batch.to(device)
          y_batch = y_batch.to(device)
          prediction = model(x_batch)[:, 0]
          is_prediction_correct = ((prediction>=0.5).float() == y_batch).float()
          testing_accuracy += is_prediction_correct.sum().cpu()

  testing_accuracy /= len(testing_data_loader.dataset)
  print(f'Test accuracy: {testing_accuracy:.4f}')