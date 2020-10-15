import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class ConstractiveLoss(nn.Module):

    def __init__(self,margin =2.0,dist_flag='l2'):
        super(ConstractiveLoss, self).__init__()
        self.margin = margin
        self.dist_flag = dist_flag
        self.algha = 0.1

    def various_distance(self,out_vec_t0,out_vec_t1):

        if self.dist_flag == 'l2':
            distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=2)
        if self.dist_flag == 'l1':
            distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=1)
        if self.dist_flag == 'cos':
            similarity = F.cosine_similarity(out_vec_t0,out_vec_t1)
            distance = 1 - 2 * similarity/np.pi
        return distance

    def forward(self,out_vec_t0,out_vec_t1,label):

        distance = self.various_distance(out_vec_t0,out_vec_t1)
        #distance = 1 - F.cosine_similarity(out_vec_t0,out_vec_t1)
        #constractive_loss = torch.sum((1-label)* self.algha * torch.pow(distance,2) + \
         #                   label * (1- self.algha) * torch.pow(torch.clamp(self.margin - distance, min=0.0),2))

        constractive_loss = torch.sum((1-label)* torch.pow(distance,2) + \
                                      label * torch.pow(torch.clamp(self.margin - distance, min=0.0),2))
        return constractive_loss

class ContrastiveLossWithOHEM(nn.Module):
    def __init__(self,margin=2.0):
        super(ContrastiveLossWithOHEM, self).__init__()
        self.margin = margin
        self.alpha = 0.3

    def forward(self, out_vec_t0,out_vec_t1,label):

        distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=2)
        similar_pair = torch.pow(1 - torch.exp(-torch.pow(distance, 2)), 2) * torch.pow(distance, 2)
        dissimilar_pair = torch.pow(torch.exp(-torch.pow(distance, 2)), 2) * torch.pow(
            torch.clamp(self.margin - distance, min=0.0), 2)
        cl_ohem = self.alpha * torch.sum((1-label) * similar_pair + (1-self.alpha) * label * dissimilar_pair)
        return cl_ohem

class ConstractiveMaskLoss(nn.Module):

    def __init__(self,thresh_flag=False,hinge_thresh=0.0,dist_flag='l2',OHEM=True):
        super(ConstractiveMaskLoss, self).__init__()
        self.sample_constractive_loss = ContrastiveLossWithOHEM(margin=2.0)

    def forward(self,out_t0,out_t1,ground_truth):

        #out_t0 = out_t0.permute(0,2,3,1)
        n,c,h,w = out_t0.data.shape
        out_t0_rz = torch.transpose(out_t0.view(c,h*w),1,0)
        out_t1_rz = torch.transpose(out_t1.view(c,h*w),1,0)
        gt_tensor = torch.from_numpy(np.array(ground_truth.data.cpu().numpy(),np.float32))
        gt_rz = Variable(torch.transpose(gt_tensor.view(1, h * w), 1, 0)).cuda()
        #gt_rz = Variable(torch.transpose(ground_truth.view(1,h*w),1,0))
        loss = self.sample_constractive_loss(out_t0_rz,out_t1_rz,gt_rz)
        return loss
