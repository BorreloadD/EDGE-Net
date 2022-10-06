import os
from torch.utils.data import DataLoader
from lib.dataset import Data
import torch.nn.functional as F
import torch
import cv2
import time
from net import edg_net
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    model_path='C:/Users/lengzhu/Desktop/code/wby-net-ver3.5/model_path/SimFs_path/epoch_200.pth'
    out_path = './output'
    data  = Data(root='D:/dataset/RGB-T/VT5000_clear/Test',mode='test')
    loader = DataLoader(data, batch_size=1,shuffle=False)
    net = edg_net().cuda()
    print('loading model from %s...' % model_path)
    net.load_state_dict(torch.load(model_path))
    if not os.path.exists(out_path): os.mkdir(out_path)
    time_s = time.time()
    img_num = len(loader)
    net.eval()
    with torch.no_grad():
        for rgb, t, _ , (H, W), name in loader:
            name = name[0].split('\\')[-1]
            print(name)
            S,F2_,F3_,F4_,ATTedg = net(rgb.cuda().float(), t.cuda().float())
            S = F.interpolate(S, size=(H, W), mode='bilinear',align_corners=True)
            pred = np.squeeze(torch.sigmoid(S).cpu().data.numpy())
            cv2.imwrite(os.path.join(out_path, name[:-4] + '.png'), 255 * pred)
    time_e = time.time()
    print('speed: %f FPS' % (img_num / (time_e - time_s)))