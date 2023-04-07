# VTS (CSQ with ViT Backbone - ICME 2022)
# paper [Vision Transformer Hashing for Image Retrieval, ICME 2022](https://arxiv.org/pdf/2109.12564.pdf)
# CSQ basecode considered from https://github.com/swuxyj/DeepHash-pytorch

from utils.tools import *
from conformer.model import  Conformer

import argparse
import os
import random
import torch
import torch.optim as optim
import time
import numpy as np
from scipy.linalg import hadamard
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.tensorboard import SummaryWriter
summaryWriter = SummaryWriter()


def get_config():
    config = {
        "dataset": "cifar",
        "alpha":0.0001,
        "beta":0.0001,
        "gamma":0.0001,     
        "lambda": 0.0001,    
        "data_path":"/root/autodl-tmp/imagenet/",        
        "net_print": "ConFormer", "model_type": "ConFormer", "pretrained_dir": "preload/Conformer_base_patch16.pth",      
        "bit_list": [64],
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-5}},
        "device": torch.device("cuda"), "save_path": "Checkpoints_Results",
        "epoch": 200, "test_map": 30, "batch_size": 1, "resize_size": 256, "crop_size": 224,
        "info": "Class Loss + Center LOSS + Q_Loss + Cauchy Loss",
        "checkpoint":False,
        "loadpre":True
    }
    config = config_dataset(config)
    return config

def train_val(config, bit):

    start_epoch = 1
    Best_mAP = 0
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    
    num_classes = config["n_class"]
    hash_bit = bit


    net = Conformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=12,
                        num_heads=9, mlp_ratio=4, qkv_bias=True, num_classes = 1000,hash_length = hash_bit)

    net = net.to(device)

    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])
    best_path = os.path.join(config["save_path"], config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(bit) + "-BestModel.pt")
    trained_path = os.path.join(config["save_path"], config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(bit) + "-IntermediateModel.pt")
    results_path = os.path.join(config["save_path"], config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(bit) + ".txt")
    f = open(results_path, 'a')
    
    if(config["checkpoint"]):
        if os.path.exists(trained_path):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(trained_path)
            net.load_state_dict(checkpoint['net'])
            Best_mAP = checkpoint['Best_mAP']
            start_epoch = checkpoint['epoch'] + 1

    if(config["loadpre"]):  
        print('==> Loading from pretrained model..')
        # net.load_state_dict(torch.load(config['pretrained_dir']),strict=False)
        state_dict = torch.load(config['pretrained_dir'])
        # 跳过或删除不需要加载的层
        state_dict.pop('trans_cls_head.weight')
        state_dict.pop('trans_cls_head.bias')
        state_dict.pop('conv_cls_head.weight')
        state_dict.pop('conv_cls_head.bias')
        # 加载权重
        net.load_state_dict(state_dict, strict=False)
    
    # 需要训练的层名称
    train_layers = {'hash_layer_trans', 'hash_layer_conv', 'conv_cls_head', 'trans_cls_head'}

    # 遍历模型的所有参数
    for name, param in net.named_parameters():
        # 检查当前参数是否属于需要训练的层
        if any(layer_name in name for layer_name in train_layers):
            # 如果是，将其requires_grad设置为True
            param.requires_grad = True
        else:
            # 如果不是，将其requires_grad设置为False
            param.requires_grad = False
    
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    print("Cau Loss")
    criterion = MyLoss(config, bit)

    for epoch in range(start_epoch, config["epoch"]+1):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s-%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], config["net_print"], epoch, config["epoch"], current_time, bit, config["dataset"]), end="")
        net.train()
        train_loss = 0
        for image, label, ind in train_loader:
            # print("###shape:",image.shape)
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            cls_out,u = net(image)
            # cls_out: [conv_cls, tran_cls] u: hash_bit
            loss ,loss_c,center_loss,quantization_loss,loss_d = criterion(u,cls_out,label.float(), ind, config)
            
            summaryWriter.add_scalar("center_loss",center_loss.item(),epoch)
            summaryWriter.add_scalar("Class",loss_c.item(),epoch)
            summaryWriter.add_scalar("cauchy",loss_d.item(),epoch)
            summaryWriter.add_scalar("quantization_loss",quantization_loss.mean().item(),epoch) 
            
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))
        summaryWriter.add_scalar("training_loss",train_loss, epoch)
        f.write('Train | Epoch: %d | Loss: %.3f\n' % (epoch, train_loss))

        # if (epoch) % config["test_map"] == 0:
            # print("calculating test binary code......")
        tst_binary, tst_label = compute_result(test_loader, net, device=device)

        # print("calculating dataset binary code.......")\
        trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

        # print("calculating map.......")
        mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                            config["topK"], summaryWriter, epoch)
        
        print("map",mAP)
        if mAP > Best_mAP:
            Best_mAP = mAP
            P, R = pr_curve(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy())
            print(f'Precision Recall Curve data:\n"CCQC":[{P},{R}],')
            f.write('PR | Epoch %d | ' % (epoch))
            for PR in range(len(P)):
                f.write('%.5f %.5f ' % (P[PR], R[PR]))
            f.write('\n')
        
            print("Saving in ", config["save_path"])
            state = {
                'net': net.state_dict(),
                'Best_mAP': Best_mAP,
                'epoch': epoch,
            }
            torch.save(state, best_path)
        print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
            config["info"], epoch, bit, config["dataset"], mAP, Best_mAP))
        f.write('Test | Epoch %d | MAP: %.3f | Best MAP: %.3f\n'
            % (epoch, mAP, Best_mAP))
        print(config)

        state = {
            'net': net.state_dict(),
            'Best_mAP': Best_mAP,
            'epoch': epoch,
        }
        torch.save(state, trained_path)
        '''state = {
            'net': net.state_dict(),
            'Best_mAP': Best_mAP,
            'epoch': epoch,
        }
        torch.save(state, trained_path)'''
    f.close()



# Center LOSS +  Class Loss + gama * Q_Loss  + (Cauchy Loss + lambda1 * quantization_loss)
# center_loss +  loss_c + self.gamma * Q_loss  + loss_d
class MyLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(MyLoss, self).__init__()
        self.is_single_label = config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}
        self.hash_targets = self.get_hash_targets(config["n_class"], bit).to(config["device"])
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(config["device"])
        self.criterion = torch.nn.BCELoss().to(config["device"])

        self.gamma = config["gamma"]
        self.lambda1 = config["lambda"]
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.K = bit
        self.one = torch.ones((config["batch_size"], bit)).to(config["device"])


    def d(self, hi, hj):
        inner_product = hi @ hj.t()
        norm = hi.pow(2).sum(dim=1, keepdim=True).pow(0.5) @ hj.pow(2).sum(dim=1, keepdim=True).pow(0.5).t()
        cos = inner_product / norm.clamp(min=0.0001)
        # formula 6
        return (1 - cos.clamp(max=0.99)) * self.K / 2


    def forward(self, u, class_out, y, ind, config):
        u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))

        Q_loss = (u.abs() - 1).pow(2).mean()

        loss_Cross = torch.nn.CrossEntropyLoss()
        #  ===============================================
        # Class Loss  
        # one-hot to labels
        lables = torch.argmax(y,dim=1)
        # loss_c = loss_Cross(class_out,lables)
        loss_list = [loss_Cross(o, lables) / len(class_out)  for o in class_out]
        loss_c = sum(loss_list)
        #===================================================
        #DCH
        s = (y @ y.t() > 0).float()

        if (1 - s).sum() != 0 and s.sum() != 0:
            # formula 2
            positive_w = s * s.numel() / s.sum()
            negative_w = (1 - s) * s.numel() / (1 - s).sum()
            w = positive_w + negative_w
        else:
            # maybe |S1|==0 or |S2|==0
            w = 1

        d_hi_hj = self.d(u, u)
        # formula 8
        cauchy_loss = w * (s * torch.log(d_hi_hj / self.gamma) + torch.log(1 + self.gamma / d_hi_hj))
        # formula 9
        quantization_loss = torch.log(1 + self.d(u.abs(), self.one) / self.gamma)
        # formula 7
        loss_d = cauchy_loss.mean() 

        #====================================================
        # summaryWriter.add_scalar("center_loss",center_loss.item(),self.epoch)
        # summaryWriter.add_scalar("Class",loss_c.item(),self.epoch)
        # summaryWriter.add_scalar("cauchy",loss_d.item(),self.epoch)
        # summaryWriter.add_scalar("quantization_loss",quantization_loss.mean().item(),self.epoch) 

        sum_loss = loss_c +  center_loss + self.alpha * loss_d + self.beta * quantization_loss.mean()
        return  sum_loss, loss_c,center_loss,quantization_loss.mean(),loss_d

    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.hash_targets[y.argmax(axis=1)]
        else:
            # to get sign no need to use mean, use sum here
            center_sum = y @ self.hash_targets
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center
    
    # use algorithm 1 to generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                # choose min(c) in the range of K/4 to K/3
                # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
                # but it is hard when bit is  small
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets
    
# class CSQLoss(torch.nn.Module):
#     def __init__(self, config, bit):
#         super(CSQLoss, self).__init__()
#         self.is_single_label = config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}
#         self.hash_targets = self.get_hash_targets(config["n_class"], bit).to(config["device"])
#         self.multi_label_random_center = torch.randint(2, (bit,)).float().to(config["device"])
#         self.criterion = torch.nn.BCELoss().to(config["device"])

#     def forward(self, u, class_out, y, ind, config):
#         u = u.tanh()
#         hash_center = self.label2center(y)
#         center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))

#         Q_loss = (u.abs() - 1).pow(2).mean()

#         loss_Cross = torch.nn.CrossEntropyLoss()
#         # one-hot to labels
#         lables = torch.argmax(y,dim=1)
#         loss_c = loss_Cross(class_out,lables)  # 分类的交叉熵损失

#         return center_loss + config["lambda"] * Q_loss + loss_c

#     def label2center(self, y):
#         if self.is_single_label:
#             hash_center = self.hash_targets[y.argmax(axis=1)]
#         else:
#             # to get sign no need to use mean, use sum here
#             center_sum = y @ self.hash_targets
#             random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
#             center_sum[center_sum == 0] = random_center[center_sum == 0]
#             hash_center = 2 * (center_sum > 0).float() - 1
#         return hash_center
    
#     # use algorithm 1 to generate hash centers
#     def get_hash_targets(self, n_class, bit):
#         H_K = hadamard(bit)
#         H_2K = np.concatenate((H_K, -H_K), 0)
#         hash_targets = torch.from_numpy(H_2K[:n_class]).float()

#         if H_2K.shape[0] < n_class:
#             hash_targets.resize_(n_class, bit)
#             for k in range(20):
#                 for index in range(H_2K.shape[0], n_class):
#                     ones = torch.ones(bit)
#                     # Bernouli distribution
#                     sa = random.sample(list(range(bit)), bit // 2)
#                     ones[sa] = -1
#                     hash_targets[index] = ones
#                 # to find average/min  pairwise distance
#                 c = []
#                 for i in range(n_class):
#                     for j in range(n_class):
#                         if i < j:
#                             TF = sum(hash_targets[i] != hash_targets[j])
#                             c.append(TF)
#                 c = np.array(c)

#                 # choose min(c) in the range of K/4 to K/3
#                 # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
#                 # but it is hard when bit is  small
#                 if c.min() > bit / 4 and c.mean() >= bit / 2:
#                     print(c.min(), c.mean())
#                     break
#         return hash_targets

# class CSQLoss(torch.nn.Module):
#     def __init__(self, config, bit):
#         super(CSQLoss, self).__init__()
#         self.is_single_label = config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}
#         self.hash_targets = self.get_hash_targets(config["n_class"], bit).to(config["device"])
#         self.multi_label_random_center = torch.randint(2, (bit,)).float().to(config["device"])
#         self.criterion = torch.nn.BCELoss().to(config["device"])

#     def forward(self, u, y, ind, config):
#         u = u.tanh()
#         hash_center = self.label2center(y)
#         center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))

#         Q_loss = (u.abs() - 1).pow(2).mean()
#         return center_loss + config["lambda"] * Q_loss

#     def label2center(self, y):
#         if self.is_single_label:
#             hash_center = self.hash_targets[y.argmax(axis=1)]
#         else:
#             # to get sign no need to use mean, use sum here
#             center_sum = y @ self.hash_targets
#             random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
#             center_sum[center_sum == 0] = random_center[center_sum == 0]
#             hash_center = 2 * (center_sum > 0).float() - 1
#         return hash_center

#     # use algorithm 1 to generate hash centers
#     def get_hash_targets(self, n_class, bit):
#         H_K = hadamard(bit)
#         H_2K = np.concatenate((H_K, -H_K), 0)
#         hash_targets = torch.from_numpy(H_2K[:n_class]).float()

#         if H_2K.shape[0] < n_class:
#             hash_targets.resize_(n_class, bit)
#             for k in range(20):
#                 for index in range(H_2K.shape[0], n_class):
#                     ones = torch.ones(bit)
#                     # Bernouli distribution
#                     sa = random.sample(list(range(bit)), bit // 2)
#                     ones[sa] = -1
#                     hash_targets[index] = ones
#                 # to find average/min  pairwise distance
#                 c = []
#                 for i in range(n_class):
#                     for j in range(n_class):
#                         if i < j:
#                             TF = sum(hash_targets[i] != hash_targets[j])
#                             c.append(TF)
#                 c = np.array(c)

#                 # choose min(c) in the range of K/4 to K/3
#                 # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
#                 # but it is hard when bit is  small
#                 if c.min() > bit / 4 and c.mean() >= bit / 2:
#                     print(c.min(), c.mean())
#                     break
#         return hash_targets

import datetime


if __name__ == "__main__":
    config = get_config()
    print(config)
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d %H:%M:%S")

    # for config['alpha'] in [10,1,0.1,0.01]:
    #     for config['beta'] in [1,0.1,0.01,0.001]:
    #         for bit in config["bit_list"]:
    #             tenlogpath = "./logs/"+ config['net_print']+ "_"+str(bit)+ "_" +str(now)+ "_"+ config['dataset'] + "["+str(config['alpha']) + " "+ str(config['beta'])  + "]"+ "/"
    #             if not os.path.exists(tenlogpath):
    #                 os.makedirs(tenlogpath)
    #             summaryWriter = SummaryWriter(tenlogpath)
    #             train_val(config, bit)
    
    # 1 作业版
    for config['alpha'] in [0.0001]:
        for config['beta'] in [0.0001]:
            for bit in config["bit_list"]:
                tenlogpath = "./logs/"+ config['net_print']+ "_"+str(bit)+ "_" +str(now)+ "_"+ config['dataset'] + "["+str(config['alpha']) + " "+ str(config['beta'])  + "]"+ "/"
                if not os.path.exists(tenlogpath):
                    os.makedirs(tenlogpath)
                summaryWriter = SummaryWriter(tenlogpath)
                train_val(config, bit)

    # #  2 029
    # for config['alpha'] in [0.0001,0.001,0.01,0.1,1]:
    #     for config['beta'] in [0.001]:
    #         for bit in config["bit_list"]:
    #             tenlogpath = "./logs/"+ config['net_print']+ "_"+str(bit)+ "_" +str(now)+ "_"+ config['dataset'] + "["+str(config['alpha']) + " "+ str(config['beta'])  + "]"+ "/"
    #             if not os.path.exists(tenlogpath):
    #                 os.makedirs(tenlogpath)
    #             summaryWriter = SummaryWriter(tenlogpath)
    #             train_val(config, bit)

    # # 3 002
    # for config['alpha'] in [0.0001,0.001,0.01,0.1,1]:
    #     for config['beta'] in [0.01]:
    #         for bit in config["bit_list"]:
    #             tenlogpath = "./logs/"+ config['net_print']+ "_"+str(bit)+ "_" +str(now)+ "_"+ config['dataset'] + "["+str(config['alpha']) + " "+ str(config['beta'])  + "]"+ "/"
    #             if not os.path.exists(tenlogpath):
    #                 os.makedirs(tenlogpath)
    #             summaryWriter = SummaryWriter(tenlogpath)
    #             train_val(config, bit)

    # # 4 012
    # for config['alpha'] in [0.0001,0.001,0.01,0.1,1]:
    #     for config['beta'] in [0.1]:
    #         for bit in config["bit_list"]:
    #             tenlogpath = "./logs/"+ config['net_print']+ "_"+str(bit)+ "_" +str(now)+ "_"+ config['dataset'] + "["+str(config['alpha']) + " "+ str(config['beta'])  + "]"+ "/"
    #             if not os.path.exists(tenlogpath):
    #                 os.makedirs(tenlogpath)
    #             summaryWriter = SummaryWriter(tenlogpath)
    #             train_val(config, bit)

    # # 5 021
    # for config['alpha'] in [0.0001,0.001,0.01,0.1,1]:
    #     for config['beta'] in [1]:
    #         for bit in config["bit_list"]:
    #             tenlogpath = "./logs/"+ config['net_print']+ "_"+str(bit)+ "_" +str(now)+ "_"+ config['dataset'] + "["+str(config['alpha']) + " "+ str(config['beta'])  + "]"+ "/"
    #             if not os.path.exists(tenlogpath):
    #                 os.makedirs(tenlogpath)
    #             summaryWriter = SummaryWriter(tenlogpath)
    #             train_val(config, bit)