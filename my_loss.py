import torch
import torch.nn as nn
import numpy as np
import random
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

class My_Loss(nn.Module):
    def __init__(self, config,bit):
        super().__init__()
        self.is_single_label = config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}
        self.target_vectors = self.get_target_vectors(config["n_class"], int(bit/2), config["p"]).to(config["device"])
        self.multi_label_random_center = torch.randint(2, (int(bit/2),)).float().to(config["device"])
        self.m = config["m"]
        self.U = torch.zeros(config["num_train"], int(bit/2) ).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

        self.num_classes = config["n_class"]
        self.hash_code_length = bit
        self.alph = config['alpha']
        self.beta = config['beta']
        self.gamm = config["gamma"]
        self.classify_loss_fun = torch.nn.CrossEntropyLoss()

    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.target_vectors[y.argmax(axis=1)]
        else:
            # for multi label, use the same strategy as CSQ
            center_sum = y @ self.target_vectors
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    # Random Assignments of Target Vectors
    def get_target_vectors(self, n_class, bit, p=0.5):
        target_vectors = torch.zeros(n_class, bit)
        for k in range(20):
            for index in range(n_class):
                ones = torch.ones(bit)
                sa = random.sample(list(range(bit)), int(bit * p))
                ones[sa] = -1
                target_vectors[index] = ones
        return target_vectors

    # Adaptive Updating
    def update_target_vectors(self):
        self.U = (self.U.abs() > self.m).float() * self.U.sign()
        self.target_vectors = (self.Y.t() @ self.U).sign()

    def balanced_prob_loss(self,X):
        """
        Calculate the equilibrium probability loss function
        Parameters:
        X: A two-dimensional numpy array representing a data set with one hash code per row and one bit of the hash code per column, with the value -1 or 1.
        Return value:
        loss: Indicates the value of the balanced probability loss function.
        """
        bit = X.shape[1]  
        S = X.shape[0]   
        X = torch.sign(X)
        P_minus_1 = torch.sum(X == -1) / (bit * S)  # -1出现概率
        P_1 = torch.sum(X == 1) / (bit * S)         # 1出现概率
        b_loss = torch.abs(-P_minus_1 * torch.log2(P_minus_1) + P_1 * torch.log2(P_1))
        return b_loss



    def forward(self, hash_out, cls_out,target,ind):
        lables = torch.argmax(target,dim=1)
        cls_loss = 0.5 * self.classify_loss_fun(cls_out[0],lables) 
        
        cls_loss = cls_loss +  0.5 *  self.classify_loss_fun(cls_out[1],lables)


        self.U[ind, :] = hash_out[0].data
        self.Y[ind, :] = target.float()
        t = self.label2center(target)
        polarization_loss_conv = (self.m - hash_out[0] * t).clamp(0).mean()

        self.U[ind, :] = hash_out[1].data
        self.Y[ind, :] = target.float()
        t = self.label2center(target)
        polarization_loss_trans = (self.m - hash_out[1] * t).clamp(0).mean()
        
        full_hash = torch.cat([hash_out[0], hash_out[1]], dim=1)
        balanced_loss = self.balanced_prob_loss(full_hash)
                
        loss = cls_loss + self.alph * (polarization_loss_conv + polarization_loss_trans) + self.beta * balanced_loss 

        return loss

