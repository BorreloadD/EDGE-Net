import torch
from torch import nn
import torch.nn.functional as F
import vgg



def convblock(in_,out_,ks,st,pad):
    return nn.Sequential(
        nn.Conv2d(in_,out_,ks,st,pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )




class CA(nn.Module):
    def __init__(self,in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        self.max_weight = nn.AdaptiveMaxPool2d(1)
        self.fus = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch, 1, 1, 0),
        )
        self.c_mask = nn.Sigmoid()
    def forward(self, x):
        avg_map_c = self.avg_weight(x)
        max_map_c = self.max_weight(x)
        c_mask = self.c_mask(torch.add(self.fus(avg_map_c), self.fus(max_map_c)))
        return torch.mul(x, c_mask)






class GlobalInfo(nn.Module):
    def __init__(self):
        super(GlobalInfo, self).__init__()
        self.ca = CA(1024)  # 通道注意力
        self.de_chan = convblock(1024, 256, 3, 1, 1)  #  卷积、BN、Rule
        self.conv1x1 = nn.Conv2d(1024,512,1)
        self.b0 = nn.Conv2d(256,128,1)
        self.b1 = nn.Sequential(
            nn.AdaptiveMaxPool2d(3),
            nn.Conv2d(256, 128, 1),
            nn.Conv2d(128, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(128, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(128, 128, 3, padding=3, dilation=3),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.AdaptiveMaxPool2d(5),
            nn.Conv2d(256, 128, 1),
            nn.Conv2d(128, 128, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(128, 128, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(128, 128, 3, padding=5, dilation=5),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.AdaptiveMaxPool2d(7),
            nn.Conv2d(256, 128, 1),
            nn.Conv2d(128, 128, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(128, 128, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(128, 128, 3, padding=7, dilation=7),
            nn.ReLU(inplace=True)
        )
        self.fus = convblock(768, 512, 3, 1, 1)  # 128*4+256=768

    def forward(self, rgb, t):
        x_size = rgb.size()[2:]
        x = self.ca(torch.cat((rgb, t), 1))  #1024
        x_ = self.de_chan(x) # 256
        b0 = F.interpolate(self.b0(x_), x_size, mode='bilinear', align_corners=True)
        b1 = F.interpolate(self.b1(x_), x_size, mode='bilinear', align_corners=True)
        b2 = F.interpolate(self.b2(x_), x_size, mode='bilinear', align_corners=True)
        b3 = F.interpolate(self.b3(x_), x_size, mode='bilinear', align_corners=True)
        out_ = self.conv1x1(x)
        out = self.fus(torch.cat((b0, b1, b2, b3, x_), 1))  # 512
        out = torch.add(out,out_)

        return out    #  512


class Edg_gt(nn.Module):
    def __init__(self):
        super(Edg_gt, self).__init__()
        self.de_chan_1 = convblock(448, 128, 3, 1, 1)
        self.de_chan_2 = convblock(128, 128, 3, 1, 1)
        self.de_chan_3 = convblock(128, 128, 3, 1, 1)
        self.de_chan_4 = convblock(128, 128, 3, 1, 1)
        self.de_chan_5 = convblock(128, 128, 3, 1, 1)
        self.conv3x3 = nn.Conv2d(128,128,3,padding=1)
        self.conv1x1 = nn.Conv2d(128,64,1)


    def forward(self,x1,x2,x3):
        x2 = F.interpolate(x2,size=352,mode='bilinear',align_corners=True)
        x3 = F.interpolate(x3,size=352,mode='bilinear',align_corners=True)
        x = torch.cat((x1,x2,x3),dim=1)
        pre = self.de_chan_1(x)
        pre = self.de_chan_2(pre)
        pre = self.de_chan_3(pre)
        # pre = self.de_chan_4(pre)
        # pre = self.de_chan_5(pre)
        pre = self.conv3x3(pre)
        pre = self.conv1x1(pre)
        pre = F.softmax(pre,dim=1)

        return pre,x

class Decoder4(nn.Module):
    def __init__(self):
        super(Decoder4, self).__init__()
        self.conv1536_512 = nn.Conv2d(1536,512,1)
        self.conv3x3 = nn.Conv2d(512,512,3,padding=1)
        self.BN = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        self.ca = CA(512)

    def forward(self,rgb,T,global_info):
        rgb = F.interpolate(self.ca(rgb),size=352,mode='bilinear',align_corners=True)   # 512
        T = F.interpolate(self.ca(T),size=352,mode='bilinear',align_corners=True) # 512
        global_info = F.interpolate(global_info, size=352, mode='bilinear', align_corners=True)
        x = torch.cat((rgb,T,global_info),dim=1)   #  1536
        x = self.conv1536_512(x)
        x = self.relu(self.BN(self.conv3x3(x)))

        return x    # 512


class Decoder3(nn.Module):
    def __init__(self):
        super(Decoder3, self).__init__()
        self.conv512_512 = nn.Conv2d(512,512,1)
        self.conv3x3 = nn.Conv2d(512, 512, 3, padding=1)
        self.BN = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        self.ca = CA(256)

    def forward(self, rgb, T, global_info,F4):
        rgb = F.interpolate(self.ca(rgb), size=352, mode='bilinear', align_corners=True)  # 256
        T = F.interpolate(self.ca(T), size=352, mode='bilinear', align_corners=True)  # 256
        global_info = F.interpolate(global_info, size=352, mode='bilinear', align_corners=True)
        global_info = self.conv3x3(global_info)
        F4 = self.conv3x3(F4)
        F4 = self.conv3x3(F4)
        x = torch.cat((rgb,T),dim=1)+F4+global_info
        x = self.conv512_512(x)
        x = self.relu(self.BN(self.conv3x3(x)))

        return x  # 512


class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.conv512_256 = nn.Conv2d(512,256,1)
        self.conv256_64 = nn.Conv2d(256,64,1)
        self.conv3x3 = nn.Conv2d(64, 64, 3, padding=1)
        self.BN = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.ca = CA(128)
        self.conv3x3_256 = nn.Conv2d(256,256,3,padding=1)

    def forward(self, rgb, T, global_info,F3):
        rgb = F.interpolate(self.ca(rgb), size=352, mode='bilinear', align_corners=True)  # 128
        T = F.interpolate(self.ca(T), size=352, mode='bilinear', align_corners=True)  # 128
        global_info = F.interpolate(global_info, size=352, mode='bilinear', align_corners=True)
        global_info = self.conv512_256(global_info)
        F3 = self.conv512_256(F3)
        F3 = self.conv3x3_256(F3)
        x = torch.cat((rgb,T),dim=1)+F3+global_info
        x = self.conv256_64(x)
        x = self.relu(self.BN(self.conv3x3(x)))

        return x  # 64








class edg_net(nn.Module):
    def __init__(self):
        super(edg_net, self).__init__()
        self.edg_extract = Edg_gt()
        self.rgb_net= vgg.a_vgg16()
        self.t_net= vgg.a_vgg16()
        self.global_info = GlobalInfo()
        self.decoder4 = Decoder4()
        self.decoder3 = Decoder3()
        self.decoder2 = Decoder2()
        self.conv128_64 = nn.Conv2d(128,64,1)
        self.conv64_1 = nn.Conv2d(64,1,1)
        self.conv512_1 = nn.Conv2d(512,1,1)
        self.conv448_64 = nn.Conv2d(448,64,1)
        self.conv3x3 = nn.Conv2d(128,128,3,padding=1)
        self.conv128_1 = nn.Conv2d(128,1,1)
        self.ca_1 = CA(64)

    def forward(self,rgb,T):
        rgb_f = self.rgb_net(rgb)
        t_f = self.t_net(T)
        global_info = self.global_info(rgb_f[4],t_f[4])   # 512 1/16
        ATTedg ,EDG_F= self.edg_extract(rgb_f[0],rgb_f[1],rgb_f[2])   #  64
        ATTedg_ = self.conv64_1(ATTedg)
        F4 = self.decoder4.forward(rgb_f[3],t_f[3],global_info)     #  1,512,352,352
        F3 = self.decoder3.forward(rgb_f[2],t_f[2],global_info,F4)  #  1,512,352,352
        F2 = self.decoder2.forward(rgb_f[1],t_f[1],global_info,F3)  #  1,64,352,352
        #  F2 和 ATTedg 都是 1, 64, 352, 352
        EDG_CA = self.ca_1(ATTedg)
        EDG_F = self.conv448_64(EDG_F)
        ATT = torch.mul(EDG_F,EDG_CA)+EDG_F
        S = torch.cat((ATT,F2),dim=1)
        S = self.conv3x3(S)
        S = self.conv128_1(S)
        F2_ = self.conv64_1(F2)
        F3_ = self.conv512_1(F3)
        F4_ = self.conv512_1(F4)
        # F2_ = F2_.squeeze(0)
        # F3_ = F3_.squeeze(0)
        # F4_ = F4_.squeeze(0)

        return S,F2_,F3_,F4_,ATTedg_

    def load_pretrained_model(self):
        st=torch.load("vgg16.pth")
        st2={}
        for key in st.keys():
            st2['base.'+key]=st[key]
        self.rgb_net.load_state_dict(st2)
        self.t_net.load_state_dict(st2)
        print('loading pretrained model success!')









