import torch
import torchvision
from torchvision import transforms

def get_data_loader(train_miniImagenet_path, batch_size):
    
    train_transforms = transforms.Compose([
              transforms.Resize((224, 224)),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
                             ])
    train_dataset = torchvision.datasets.ImageFolder(root = train_miniImagenet_path, transform = train_transforms)
    base_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    
    return base_loader
    

def unlabelled_loader(train_miniImagenet_path, batch_size):
    
    train_transforms = transforms.Compose([
              transforms.Resize((224, 224)),
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
                             ])
    train_dataset = torchvision.datasets.ImageFolder(root = train_miniImagenet_path, transform = train_transforms)
    base_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    
    return base_loader