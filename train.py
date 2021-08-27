import torch
import torchvision
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.optim
from torch.optim import lr_scheduler
import time
import os
import glob

import configs
import backbone

from io_utils import parse_args, get_resume_file  
from datasets import miniImageNet_few_shot, DTD_few_shot


def train(base_loader, model, start_epoch, stop_epoch, save_dir):
    #num_epochs = 1000
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.0001)
    num_epochs = stop_epoch - start_epoch
    num_batches = 0
    for epoch in range(num_epochs):
        
        model.train()
        train_accuracy = 0.0
        train_loss = 0.0
        num_batches = 0

        for i, (images, labels) in enumerate(base_loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().data*images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            train_accuracy += int(torch.sum(prediction == labels.data))
            num_batches += 1
            print("Epoch{}__batch{}".format(epoch, num_batches))
  
        print("epoch: {}/{}".format(epoch, num_epochs))
        if (epoch % 50==0) or (epoch==stop_epoch-1):
            #utfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save(model, save_dir + '{}_resnet18.tar')
        


if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    image_size = 224
    optimizer = 'Adam'

    base_loader = miniImageNet_few_shot.get_data_loader(train_miniImagenet_path = configs.miniImageNet_path, batch_size = configs.batch_size)

    model = backbone.get_resnet18(pre_imgnet = False, num_classes = 64)
    model = model.cuda()
    save_dir =  configs.save_dir

    params.checkpoint_dir = configs.save_dir

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = configs.start_epoch
    stop_epoch = configs.stop_epoch

    model = train(base_loader, model, start_epoch, stop_epoch, save_dir)