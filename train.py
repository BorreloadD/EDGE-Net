import numpy as np
import tensorflow
import torchvision.datasets
from torch.backends import cudnn
import torchvision.models as models
from config import opt
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import cv2
from lib.dataset import Data
from lib.data_prefetcher import DataPrefetcher
from torch.utils.data import DataLoader
from net import edg_net
import torch.optim as optim











cudnn.benchmark = True  # 合理分配显存，加快训练速度


CE = torch.nn.BCEWithLogitsLoss()
L1_loss = nn.L1Loss()
step = 0
writer = SummaryWriter('my_log/summary')
best_mae = 1
best_epoch = 0
# dataset
img_root = 'D:/dataset/RGB-T/VT5000_clear/Train'
save_path = 'C:/Users/lengzhu/Desktop/code/wby-net-ver3.5/model_path/SimFs_path'
test_root = 'D:/dataset/RGB-T/VT821'
if not os.path.exists(save_path): os.mkdir(save_path)
lr = 0.001
batch_size = 3
epoch = 200
lr_dec = [101, 151]
data = Data(img_root)  # train_size=352
test_data = Data(test_root)
loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)
test_loader = DataLoader(test_data,batch_size=batch_size, shuffle=False, num_workers=1)
net = edg_net().cuda()

optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005, momentum=0.9)
iter_num = len(loader)



def train(loader, net, optimizer, epoch,save_path,lr,lr_dec):
    global step
    net.train()
    epoch_step = 0


    if epoch in lr_dec:
        lr = lr / 10
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005,
                              momentum=0.9)
        print(lr)
    prefetcher = DataPrefetcher(loader)   #  这个函数提前转换为cuda了
    rgb, t, label = prefetcher.next()
    label_1 = label[0].cpu()
    # print(label_1.shape,label.shape)   # torch.Size([1, 352, 352]) torch.Size([3, 1, 352, 352])
    gt_1 = label_1.permute(1,2,0)
    gt__1 = np.array(gt_1)
    gt__1 = (gt__1*255).astype(np.uint8)
    edg_1 = cv2.Canny(gt__1, 0, 255)
    edg_1 = edg_1.astype(np.float32)/255
    edg_1 = torch.from_numpy(edg_1).cuda()
    edg_1 = edg_1.unsqueeze(0)   #   ([1, 352, 352])

    label_2 = label[1].cpu()
    gt_2 = label_2.permute(1,2,0)
    gt__2 = np.array(gt_2)
    gt__2 = (gt__2*255).astype(np.uint8)
    edg_2 = cv2.Canny(gt__2, 0, 255)
    edg_2 = edg_2.astype(np.float32)/255
    edg_2 = torch.from_numpy(edg_2).cuda()
    edg_2 = edg_2.unsqueeze(0)

    label_3 = label[2].cpu()
    gt_3 = label_3.permute(1,2,0)
    gt__3 = np.array(gt_3)
    gt__3 = (gt__3*255).astype(np.uint8)
    edg_3 = cv2.Canny(gt__3, 0, 255)
    edg_3 = edg_3.astype(np.float32)/255
    edg_3 = torch.from_numpy(edg_3).cuda()
    edg_3 = edg_3.unsqueeze(0)

    edg = torch.cat((edg_1.unsqueeze(0),edg_2.unsqueeze(0),edg_3.unsqueeze(0)),dim=0)
    edg = edg.cuda()

    label = label.cuda()

    # print(edg.shape)
    # os.system("pause")
    r_sal_loss = 0
    net.zero_grad()
    i = 0
    # F4= net.forward(rgb,t,label,edg)
    # print(F4.shape)
    # os.system("pause")


    while rgb is not None:
        i += 1
        S,F2,F3,F4,ATTedg= net(rgb, t)
        # S = S.squeeze(0)
        # ATTedg = ATTedg.squeeze(0)
        loss_S = CE(S,label)  * 7
        loss_F2 = CE(F2,label)  * 5
        loss_F3 = CE(F3, label) * 3
        loss_F4 = CE(F4, label)
        loss_edg = L1_loss(ATTedg,edg)

        sal_loss = loss_S + loss_F2 +loss_F3 + loss_F4 + loss_edg
        r_sal_loss += sal_loss.data
        sal_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        epoch_step += 1
        if i % 100 == 0:
            print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f' % (
                epochi, 200, i, iter_num, r_sal_loss / 100))
            writer.add_scalar('Loss', sal_loss.data, global_step=step)
            grid_image = make_grid(rgb.clone().cpu().data, 1, normalize=True)
            writer.add_image('RGB', grid_image, step)
            grid_image = make_grid(label.clone().cpu().data, 1, normalize=True)
            writer.add_image('Ground_truth', grid_image, step)

            grid_image = make_grid(ATTedg.clone().cpu().data, 1, normalize=True)
            writer.add_image('ATTedg', grid_image, step)
            grid_image = make_grid(edg.clone().cpu().data, 1, normalize=True)
            writer.add_image('edg', grid_image, step)


            res1 = S[0].clone()
            res1 = res1.sigmoid().data.cpu().numpy().squeeze()
            res1 = (res1 - res1.min()) / (res1.max() - res1.min() + 1e-8)
            writer.add_image('pred', torch.tensor(res1), step, dataformats='HW')
            res2 = S[1].clone()
            res2 = res2.sigmoid().data.cpu().numpy().squeeze()
            res2 = (res2 - res2.min()) / (res2.max() - res2.min() + 1e-8)
            writer.add_image('pred', torch.tensor(res2), step, dataformats='HW')
            res3 = S[2].clone()
            res3 = res3.sigmoid().data.cpu().numpy().squeeze()
            res3 = (res3 - res3.min()) / (res3.max() - res3.min() + 1e-8)
            writer.add_image('pred', torch.tensor(res3), step, dataformats='HW')
            r_sal_loss = 0
        rgb, t, label = prefetcher.next()

    if epochi % 5 == 0:
        torch.save(net.state_dict(), '%s/epoch_%d.pth' % (save_path, epochi))
    torch.save(net.state_dict(), '%s/final.pth' % (save_path))



# def test(test_loader,net,epoch,save_path):
#     global best_mae, best_epoch
#     net.eval()
#     with torch.no_grad():
#         mae_sum = 0
#         while rgb is not None:
#             prefetcher = DataPrefetcher(loader)  # 这个函数提前转换为cuda了
#             rgb, t, label = prefetcher.next()
#
#             S,F2,F3,F4,ATTedg= net(rgb, t)
#             res = S
#             res = res.sigmoid().data.cpu().numpy().squeeze()
#             res = (res - res.min()) / (res.max() - res.min() + 1e-8)
#             mae_sum += np.sum(np.abs(res-label)*1.0/(label.shape[2]*label.shape[3]))
#             rgb, t, label = prefetcher.next()
#
#
#     mae = mae_sum/len(test_loader)
#     writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
#     print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
#     if epoch == 1:
#         best_mae = mae
#     else:
#         if mae < best_mae:
#             best_mae = mae
#             best_epoch = epoch
#             torch.save(net.state_dict(), save_path+'edgnet_epoch_best.pth')
#             print('best epoch:{}'.format(epoch))
#     logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))





if __name__ == '__main__':
    print("Start train...")
    net.load_pretrained_model()
    for epochi in range(1,epoch+1):
        train(loader, net, optimizer, epochi, save_path,lr,lr_dec)
        # test(test_loader,net, epochi, save_path)



