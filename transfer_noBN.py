import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
from itertools import combinations
import torchvision
import torch.optim as optim
import torch.backends.cudnn as cudnn
from res10_model_noaffine import *


from datasets import miniImageNet_few_shot
import configs
from methods.protonet import ProtoNet



from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot





def sbm_finetune(source_loader, target_name , num_epochs, ): 
    

    ###############################################################################################
    # load resnet18 model
    save_dir = './logs/vanilla_noBN/'    
    model = resnet10()
    #model.load_state_dict(torch.load('./logs/resnet18_imgnet.tar'))
    
    
    #model = reset_last_block(model)
    

    model.output = nn.Linear(512, 64)
    print(model)
    model.cuda()
    model.train()
    
    train_accu = []
    train_losses = []

    for epoch in range(num_epochs):
        

        
        ###############################################################################################
        loss_CE = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        ###############################################################################################
        running_loss = 0
        correct = 0
        total = 0
        train_accuracy = 0.0
        train_loss = 0.0
        num_batches = 0

        for i, (images, labels) in enumerate(base_loader):
            
            source_images = Variable(images.cuda())
            source_labels = Variable(labels.cuda())

            optimizer.zero_grad()

            outputs = model(source_images)


            ce_loss = loss_CE(outputs, source_labels)


            loss = ce_loss

            loss.backward()
            optimizer.step()
            
            
            running_loss += loss.item()
     
            _, predicted = outputs.max(1)
            total += source_labels.size(0)
            correct += predicted.eq(source_labels).sum().item()
            num_batches += 1
            print("Epoch: {} | Batch: {}".format(epoch, num_batches))
       
        train_loss=running_loss/len(base_loader)
        accu=100.*correct/total
   
        train_accu.append(accu)
        train_losses.append(train_loss)
        print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))
          
        print("epoch: {}/{}".format(epoch, num_epochs))
 
        if (epoch % 50==0):

            torch.save(model.state_dict(), save_dir + '{}_epoch{}_vanilla_noaffine.pth'.format(target_name, epoch))



if __name__ == '__main__':
    
    seed_ = 2021
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    cudnn.deterministic = True
    ##################################################################
    epochs = 601
    image_size = 224
    
    #mini_imagenet_path = '/content/miniImagenet/'
    base_loader = miniImageNet_few_shot.get_data_loader(configs.miniImageNet_path, batch_size = 128)


    ##################################################################
    pretrained_dataset = "miniImageNet"

    
     
    # replace finetine() with your own method
    sbm_finetune(base_loader, pretrained_dataset, epochs)
