
from utils.tools import *
import os
import torch
import torch.optim as optim
import time
import numpy as np
import random
import datetime
from conformer.model import  Conformer
torch.multiprocessing.set_sharing_strategy('file_system')
from my_loss import My_Loss

def get_config():
    config = {
        "dataset": "imagenet",
        "alpha":0.0001,
        "beta":0.01,
        "m": 1,
        "p": 0.5,     
        "net_print": "ConFormer", "model_type": "ConFormer", "pretrained_dir": "preload/Conformer_base_patch16.pth",   
        "bit_list": [64,32,16],
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-5}},
        "device": torch.device("cuda"), "save_path": "Checkpoints_Results",
        "epoch": 180, "test_map": 1, "batch_size": 32, "resize_size": 256, "crop_size": 224,
        "info": "DSH", "alpha": 0.1,
        "checkpoint":False,
        "loadpre":True,
        "save_path": "save/ConformerHash",
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
                        num_heads=9, mlp_ratio=4, qkv_bias=True, num_classes =num_classes,hash_length = hash_bit) # Base
    # net = Conformer(patch_size=16, channel_ratio=4, embed_dim=384, depth=12,
    #                         num_heads=6, mlp_ratio=4, qkv_bias=True,num_classes =num_classes,hash_length = hash_bit) # Small
    
#     net = Conformer(patch_size=16, channel_ratio=1, embed_dim=384, depth=12, 
#                       num_heads=6, mlp_ratio=4, qkv_bias=True, num_classes =num_classes,hash_length = hash_bit) # Tiny
    
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
        state_dict = torch.load(config['pretrained_dir'])
        state_dict.pop('trans_cls_head.weight')
        state_dict.pop('trans_cls_head.bias')
        state_dict.pop('conv_cls_head.weight')
        state_dict.pop('conv_cls_head.bias')
        net.load_state_dict(state_dict, strict=False)
    
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    criterion = My_Loss(config, bit)  

    for epoch in range(start_epoch, config["epoch"]+1):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s-%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], config["net_print"], epoch, config["epoch"], current_time, bit, config["dataset"]), end="")
        net.train()
        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            cls_out,u = net(image)
            hash_loss, quanti_loss, cls_loss, loss = criterion(u, cls_out,label.float(),ind) 
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))
        f.write('Train | Epoch: %d | Loss: %.3f\n' % (epoch, train_loss))

        if (epoch) % config["test_map"] == 0:
            # print("calculating test binary code......")
            tst_binary, tst_label = compute_result(test_loader, net, device=device)

            # print("calculating dataset binary code.......")\
            trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

            # print("calculating map.......")
            # mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
            #                  config["topK"])
            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                                config["topK"])

            if mAP > Best_mAP:
                Best_mAP = mAP
                P, R = pr_curve(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy())
                print(f'Precision Recall Curve data:\n"ConformerHash":[{P},{R}],')
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
                
                np.save(os.path.join(config["save_path"], "tst_label.npy"), tst_label.numpy())
                np.save(os.path.join(config["save_path"], "tst_binary.npy"), tst_binary.numpy())
                np.save(os.path.join(config["save_path"], "trn_binary.npy"), trn_binary.numpy())
                np.save(os.path.join(config["save_path"], "trn_label.npy"), trn_label.numpy())
                
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
    f.close()


if __name__ == "__main__":
    config = get_config()
    for bit in config["bit_list"]:
        train_val(config, bit)

