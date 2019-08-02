# Imports libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms, models
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import argparse
from collections import OrderedDict

# data dir
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    
    # Add architecture selection to parser
    parser.add_argument('--arch', 
                        type=str, 
                        help='Choose architecture from torchvision.models as str')
    
    # Add checkpoint directory to parser
    parser.add_argument('--save_dir', 
                        type=str, 
                        help='Define save directory for checkpoints as str. If not specified then model will be lost.')
    
    # Add hyperparameter tuning to parser
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='Define gradient descent learning rate as float')
    parser.add_argument('--epochs', 
                        type=int, 
                        help='Number of epochs for training as int')

    # Add GPU Option to parser
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU + Cuda for calculations')
    
    # Parse args
    args = parser.parse_args()
    return args

# Function train_transformer(train_dir) performs training transformations on a dataset
def train_transformer(train_dir):
   # Define transformation
   data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    # Load the Data
    image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
    return image_datasets

# Function test_transformer(test_dir) performs test/validation transformations on a dataset
def test_transformer(test_dir):
    # Define transformation
    testing_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    # Load the Data
    testing_datsets = datasets.ImageFolder(test_dir, transform=testing_transforms)
    return test_data

# Function data_loader(data, train=True) creates a dataloader from dataset imported
def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=32)
    return loader

# Function check_gpu(gpu_arg) make decision on using CUDA with GPU or CPU
def check_gpu(gpu_arg):
   # If gpu_arg is false then simply return the cpu device
    if not gpu_arg:
        return torch.device("cpu")
    
    # If gpu_arg then make sure to check for CUDA before assigning it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Print result
    if device == "cpu":
        print("CUDA was not found, using CPU instead.")
    return device

# primaryloader_model(architecture="vgg16") downloads model (primary) from torchvision
def primaryloader_model(architecture="vgg16"):
    model = models.vgg16(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
    param.requires_grad = False
    
    return model

# Function initial_classifier(model) creates a classifier with the corect number of input layers
def initial_classifier(model, hidden_units):
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4000)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(4000, 200)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(p = 0.15)),
                          ('fc3', nn.Linear(200, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    return classifier

# Define function for validation
def validation(model, validloader, criterion):
    valid_loss = 0
    accuracy = 0
    for images, labels in validloader:
        images = images.to(device)
        labels = labels.to(device)
        output = model.forward(images)
        valid_loss = valid_loss + criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim = 1)[1])
        accuracy = accuracy + equality.type(torch.FloatTensor).mean()
    return valid_loss, accuracy

# Function network_trainer represents the training of the network model
def network_trainer(Model, Trainloader, Testloader, Device, 
                  Criterion, Optimizer, Epochs = 3, Print_every, Steps):
    # select cuda
    model.to('cuda')
    
    for e in range (epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(dataloaders):
        steps += 1
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        optimizer.zero_grad()
        
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval()
            valid_loss = 0
            accuracy = 0
            
            for ii, (inputs2, labels2) in enumerate (validloader):
                optimizer.zero_grad()
                inputs2 = inputs2.to('cuda:0') 
                labels2 = labels2.to('cuda:0')
                model.to('cuda:0')
                with torch.no_grad():
                        outputs = model.forward(inputs2)
                        valid_loss = criterion(outputs, labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
           
            valid_loss = valid_loss / len(validloader)
            accuracy = accuracy / len(validloader)   
            
            
            print(
              "Epoch: {}/{}.. ".format(e+1, epochs),
              "Loss: {:.3f}.. ".format(running_loss/print_every),
              "Validation Loss: {:.3f}.. ".format(valid_loss),
              "Accuracy: {:.3f}".format(accuracy))
        
   
            running_loss = 0
            model.train()
        
    return model

#Function validate_model(Model, Testloader) Validation on the test datasets
def validate_model(Model, Testloader):
    correct = 0
    total = 0
    model.to('cuda:0')
    with torch.no_grad():
         for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    
# Function initial_checkpoint(Model) saves the model at a defined checkpoint
def initial_checkpoint(Model, Save_Dir, Train_data):
    # Save model at checkpoint
    if type(Save_Dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        if isdir(Save_Dir):
            # Create `class_to_idx` attribute in model
            model.class_to_idx = image_datasets.class_to_idx
            checkpoint = {'classifier' : model.classifier,
                         'epochs': args.epochs,
                          'class_to_idx': model.class_to_idx,
                          'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()
                          }

            torch.save(checkpoint, 'checkpoint.pth')
    
    
    # Get Keyword Args for Training
    args = arg_parser()
    
    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Pass transforms in, then create trainloader
    image_datasets = test_transformer(train_dir)
    validation_datasets = train_transformer(valid_dir)
    testing_datsets = train_transformer(test_dir)
    
    dataloaders = data_loader(image_datasets)
    validloader = data_loader(validation_datasets, train=False)
    testloader = data_loader(testing_datsets, train=False)
    
    # Load Model
    model = primaryloader_model(architecture="vgg16")
    
    # Build Classifier
    model.classifier = initial_classifier(model)
     
    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu);
    
    # Send model to device
    model.to(device);
    
    # Check for learnrate args
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else: learning_rate = args.learning_rate
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
   
    # Define deep learning method
    print_every = 10
    steps = 0
    
        # Train the classifier layers using backpropogation
    trained_model = network_trainer(model, trainloader, validloader, 
                                  device, criterion, optimizer, args.epochs, 
                                  print_every, steps)
    
    print("\nTraining process is now complete!!")
    
    # Quickly Validate the model
    validate_model(trained_model, testloader)
    
    # Save the model
    initial_checkpoint(trained_model, args.save_dir, image_datasets)


# Run Program
if __name__ == '__main__': main()
        

        
