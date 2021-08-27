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

from datasets import miniImageNet_few_shot
import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.protonet import ProtoNet

from io_utils import  parse_args, get_resume_file, get_best_file, get_assigned_file 

from utils import *

from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x


def clone_BN_affine(model):
    BN_statistics_list = []
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            BN_statistics_list.append(
                {'weight': layer.weight.clone(),
                 'bias': layer.bias.clone()})
    return BN_statistics_list


def clone_BN_stat(model):
    BN_statistics_list = []
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            BN_statistics_list.append(
                {'means': layer.running_mean.clone(),
                 'vars': layer.running_var.clone()})
    return BN_statistics_list


def regret_affine(source_affine, model):
    i = 0
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.bias = nn.Parameter(source_affine[i]['bias'])
            layer.weight = nn.Parameter(source_affine[i]['weight'])
            i += 1
    return model

def regret_stat(source_stat, model):
    i = 0
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.running_mean = nn.Parameter(source_stat[i]['means'])
            layer.running_var = nn.Parameter(source_stat[i]['vars'])
            i += 1
    return model

def shift_affine(source_stat, model):
    total_shift = 0
    i = 0
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            target_mean = layer.running_mean.clone()  # source state
            source_mean = source_stat[i]['means']
            shift_value = (target_mean - source_mean)
            total_shift += torch.sum(shift_value)
            # shift bias
            layer.bias = nn.Parameter((torch.rand(len(source_mean)).to(
                device) * shift_value.to(device)).to(
                   device) + layer.bias).to(device)
            i += 1
    return total_shift


def finetune(novel_loader, target_loader, n_query = 15, pretrained_dataset='miniImageNet', freeze_backbone = False, n_way = 5, n_support = 5): 
    correct = 0
    count = 0

    iter_num = len(novel_loader)

    acc_all = []

    for _, (x, y) in enumerate(novel_loader):

        ###############################################################################################
        # load pretrained model on miniImageNet
        pretrained_model = torchvision.models.resnet18(pretrained = True)

        
        ###############################################################################################

        classifier = Classifier(512, n_way)
        loss_mse = nn.MSELoss()
        loss_ce = nn.CrossEntropyLoss()
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

                source_affine = clone_BN_affine(pretrained_model)
                source_stat = clone_BN_stat(pre_trained_model)


                target_batch, _ = next(iter(target_loader))
                target_batch = target_batch.to(device)
                temp_y = pretrained_model(target_batch)
                pretrained_model = shift_affine(source_stat)

                shifted_scores = pretrained_model(x)
                
                pretrained_model = regret_affine(source_affine)

                ce_loss = loss_ce(output, y_batch)
                mse_loss = loss_mse(shifted_scores, source_scores)

                loss = ce_loss + mse_loss
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
    np.random.seed(10)
    params = parse_args('train')

    ##################################################################
    image_size = 224
    iter_num = 600

    n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 
   
    freeze_backbone = params.freeze_backbone
    ##################################################################
    pretrained_dataset = "miniImageNet"

    dataset_names = [ "EuroSAT"]
    novel_loaders = []
    unlabelled_loaders = []

    #print ("Loading ISIC")
    #datamgr             =  ISIC_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    #novel_loader        = datamgr.get_data_loader(aug =False)
    #novel_loaders.append(novel_loader)
    
    print ("Loading EuroSAT")
    datamgr             =  EuroSAT_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    novel_loader        = datamgr.get_data_loader(aug =False)
    novel_loaders.append(novel_loader)
    unlabelled_path = '/content/eurosat_unlabel'
    unlabelled_loader = miniImageNet_few_shot.unlabelled_loader(unlabelled_path, batch_size = 16)
    unlabelled_loaders.append(unlabelled_loader)
    
    #print ("Loading CropDisease")
    #datamgr             =  CropDisease_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    #novel_loader        = datamgr.get_data_loader(aug =False)
    #novel_loaders.append(novel_loader)
    
    #print ("Loading ChestX")
    #datamgr             =  Chest_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
    #novel_loader        = datamgr.get_data_loader(aug =False)
    #novel_loaders.append(novel_loader)
    
    #########################################################################
    for idx, novel_loader in enumerate(novel_loaders):
        print (dataset_names[idx])
        start_epoch = params.start_epoch
        stop_epoch = params.stop_epoch
        print (freeze_backbone)
        unlabelled_loader = unlabelled_loaders[idx]
        
        # replace finetine() with your own method
        finetune(novel_loader, unlabelled_loader, n_query = 15, pretrained_dataset=pretrained_dataset, freeze_backbone=freeze_backbone, **few_shot_params)
