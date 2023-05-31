import torchvision.transforms as transforms
import torch
import yaml
from utils import *
import time
import os
import shutil
import torch.backends.cudnn as cudnn
from test_dataloader import get_loader_test
import pickle
import argparse
from PLIPmodel import Create_PLIP_Model

def Test_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/PLIP_MRN50.pth.tar')
    parser.add_argument('--image_path', type=str, default='data/CUHK-PEDES/imgs')
    parser.add_argument('--test_path', type=str,
                        default='data/CUHK-PEDES/CUHK-PEDES-test.json',
                        help='path for test annotation json file')
# ***********************************************************************************************************************
# 设置模型backbone的类型和参数
    parser.add_argument('--plip_model', type=str, default='MResNet_BERT')
    parser.add_argument('--img_backbone', type=str, default='ModifiedResNet',
                        help="ResNet:xxx, ModifiedResNet, ViT:xxx")
    parser.add_argument('--txt_backbone', type=str, default="bert-base-uncased")
    parser.add_argument('--img_dim', type=int, default=768, help='dimension of image embedding vectors')
    parser.add_argument('--text_dim', type=int, default=768, help='dimension of text embedding vectors')
    parser.add_argument('--layers', type=list, default=[3, 4, 6, 3], help='Just for ModifiedResNet model')
    parser.add_argument('--heads', type=int, default=8, help='Just for ModifiedResNet model')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)

    # 设置超参数
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device',type=str,default="cuda:0")
    parser.add_argument('--feature_size', type=int, default=768)
    args = parser.parse_args()
    return args

def test(image_test_loader,text_test_loader, model):
    # switch to evaluate mode
    model = model.eval()
    device = next(model.parameters()).device

    qids, gids, qfeats, gfeats = [], [], [], []
    # text
    for pid, caption,mask in text_test_loader:
        caption = caption.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            text_feat = model.get_text_global_embedding(caption,mask)
        qids.append(pid.view(-1))  # flatten
        qfeats.append(text_feat)
    qids = torch.cat(qids, 0)
    qfeats = torch.cat(qfeats, 0)

    # image
    for pid, img in image_test_loader:
        img = img.to(device)
        with torch.no_grad():
            img_feat = model.get_image_embeddings(img)
        gids.append(pid.view(-1))  # flatten
        gfeats.append(img_feat)
    gids = torch.cat(gids, 0)
    gfeats = torch.cat(gfeats, 0)
    ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP = test_map(qfeats, qids, gfeats, gids)
    return ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP


def Test_main(args):
    device = args.device
    transform = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize((0.357, 0.323, 0.328),
                             (0.252, 0.242, 0.239))
    ])

    image_test_loader,text_test_loader = get_loader_test(args, transform,  args.batch_size,args.num_workers)
    model = Create_PLIP_Model(args).to(device)

    model_file = args.model_path
    print(model_file)
    if os.path.isdir(model_file):
        continue
    checkpoint = torch.load(model_file,map_location='cpu')
    model.image_encoder.load_state_dict(checkpoint["ImgEncoder_state_dict"])
    model.text_encoder.load_state_dict(checkpoint["TxtEncoder_state_dict"])
    ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP = test(image_test_loader,text_test_loader, model)
        
    print('R@1:{:.5f}  R@5:{:.5f}  R@10:{:.5f}  mAP:{:.5f}'.format(ac_t2i_top1, ac_t2i_top5, ac_t2i_top10, mAP))

import warnings
warnings.filterwarnings("ignore")
if __name__ == '__main__':
    args = Test_parse_args()
    Test_main(args)
