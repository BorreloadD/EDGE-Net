import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--testsize', type=int, default=352, help='testing dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
parser.add_argument('--T_root', type=str, default='D:/dataset/RGB-T/VT5000_clear/Train/T/', help='path of depth')
parser.add_argument('--img_root', type=str, default="D:/dataset/RGB-T/VT5000_clear/Train/RGB/", help='root of image')
parser.add_argument('--gt_root', type=str, default="D:/dataset/RGB-T/VT5000_clear/Train/GT/", help='root of gts')
parser.add_argument('--model_path', type=str, default='C:/Users/lengzhu/Desktop/code/wby-net-ver3.5/model_path/SimFs_path/', help='path of SimFs')

# 测试集
parser.add_argument('--test_image', type=str, default="D:/dataset/RGB-T/VT5000_clear/Test/RGB/", help='root of test image')
parser.add_argument('--test_gt', type=str, default="D:/dataset/RGB-T/VT5000_clear/Test/GT/", help='root of test gts')
parser.add_argument('--test_t', type=str, default="D:/dataset/RGB-T/VT5000_clear/Test/T/",help='root of test focals')

opt = parser.parse_args()
