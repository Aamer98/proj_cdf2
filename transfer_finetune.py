import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
from itertools import combinations
import torch.backends.cudnn as cudnn
from res10_model import *


import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.protonet import ProtoNet

from io_utils import parse_args, get_resume_file, get_best_file, get_assigned_file 

from utils import *

from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

def finetune(model_weights, novel_loader, n_query = 15, pretrained_dataset='miniImageNet', freeze_backbone = False, n_way = 5, n_support = 5): 
    correct = 0
    count = 0

    iter_num = len(novel_loader)
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    acc_all = []
    seed_ = 2021

    np.random.seed(seed_)
    torch.manual_seed(seed_)
    cudnn.deterministic = True

    for _, (x, y) in enumerate(novel_loader):


        ###############################################################################################
        # load pretrained model on miniImageNet
        pretrained_model = resnet10()
        pretrained_model.output = nn.Linear(512, 64)
        pretrained_model.load_state_dict(torch.load(model_weights))
        #model_path = './from_baseline_resnet18_MiniImageNet_to_EuroSAT.pth'
        #pretrained_model.load_state_dict(torch.load(model_path, map_location='cuda:0'))

        pretrained_model.output = Identity()
        ###############################################################################################

        classifier = Classifier(512, n_way)

        ###############################################################################################
        n_query = x.size(1) - n_support
        x = x.cuda()
        x_var = Variable(x)

    
        batch_size = 4
        support_size = n_way * n_support 
       
        y_a_i = Variable( torch.from_numpy( np.repeat(range( n_way ), n_support ) )).cuda() # (25,)

        x_b_i = x_var[:, n_support:,:,:,:].contiguous().view( n_way* n_query,   *x.size()[2:]) 
        x_a_i = x_var[:,:n_support,:,:,:].contiguous().view( n_way* n_support, *x.size()[2:]) # (25, 3, 224, 224)

         ###############################################################################################
        loss_fn = nn.CrossEntropyLoss().cuda()
        classifier_opt = torch.optim.SGD(classifier.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        

        if freeze_backbone is False:
            delta_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, pretrained_model.parameters()), lr = 0.01)


        pretrained_model.cuda()
        classifier.cuda()
        ###############################################################################################
        total_epoch = 100

        if freeze_backbone is False:
            pretrained_model.train()
        else:
            pretrained_model.eval()
        
        classifier.train()

        for epoch in range(total_epoch):
            rand_id = np.random.permutation(support_size)

            for j in range(0, support_size, batch_size):
                classifier_opt.zero_grad()
                if freeze_backbone is False:
                    delta_opt.zero_grad()

                #####################################
                selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).cuda()
               
                z_batch = x_a_i[selected_id]
                y_batch = y_a_i[selected_id] 
                #####################################

                output = pretrained_model(z_batch)
                output = classifier(output)
                loss = loss_fn(output, y_batch)

                #####################################
                loss.backward()

                classifier_opt.step()
                
                if freeze_backbone is False:
                    delta_opt.step()

        pretrained_model.eval()
        classifier.eval()

        output = pretrained_model(x_b_i.cuda())
        scores = classifier(output)
       
        y_query = np.repeat(range( n_way ), n_query )
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        correct_this, count_this = float(top1_correct), len(y_query)
        print (correct_this/ count_this *100)
        acc_all.append((correct_this/ count_this *100))
        
        ###############################################################################################

    acc_all  = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std  = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))



if __name__=='__main__':
    

    
    np.random.seed(2021)
    params = parse_args('train')

    ##################################################################
    image_size = 224
    iter_num = 600

    n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 
   
    freeze_backbone = params.freeze_backbone
    ##################################################################
    pretrained_dataset = "miniImageNet"

    dataset_names = "EuroSAT"
    novel_loaders = []

    #print ("Loading ISIC")
    #datamgr             =  ISIC_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    #novel_loader        = datamgr.get_data_loader(aug =False)
    #novel_loaders.append(novel_loader)
    
    print ("Loading EuroSAT")
    datamgr             =  EuroSAT_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(aug =False)
    

    #print ("Loading CropDisease")
    #datamgr             =  CropDisease_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    #novel_loader        = datamgr.get_data_loader(aug =False)
    #novel_loaders.append(novel_loader)
    
    #print ("Loading ChestX")
    #datamgr             =  Chest_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    #novel_loader        = datamgr.get_data_loader(aug =False)
    #novel_loaders.append(novel_loader)
    
    #########################################################################
    main_dir = './logs/vanilla/'

    for epch_wt in range(600, 50, -100):

        weight_dirs = 'miniImageNet_epoch{}_vanilla.pth'.format(str(epch_wt))
        print('*******************'+ main_dir + weight_dirs + '*******************')
        print (dataset_names)
        start_epoch = params.start_epoch
        stop_epoch = params.stop_epoch
        print (freeze_backbone)
        
        # replace finetine() with your own method
        finetune(main_dir + weight_dirs, novel_loader, n_query = 15, pretrained_dataset=pretrained_dataset, freeze_backbone=freeze_backbone, **few_shot_params)