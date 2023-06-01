import torchvision.transforms as transforms
from utils import *
import os
import shutil
from dataloader import get_loader_test
import argparse
from model import Create_model
import warnings
warnings.filterwarnings("ignore")

def Test_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='log/CUHK-CMPMC')
    parser.add_argument('--image_dir', type=str, default='data/CUHK-PEDES/')
    parser.add_argument('--caption_path', type=str,
                        default='data/CUHK-PEDES/CUHK-PEDES-test.json',
                        help='path for test annotation json file')
    parser.add_argument('--best_dir', type=str, default='log/CUHK-CMPMC')
    parser.add_argument('--flip_eval', type=bool, default=False)
    parser.add_argument('--mean', default=[0.357, 0.323, 0.328], type=list)
    parser.add_argument('--std', default=[0.252, 0.242, 0.239], type=list)
# ***********************************************************************************************************************
# 设置模型backbone的类型和参数
    parser.add_argument('--img_backbone', type=str, default='ModifiedResNet',
                        help="ResNet:xxx, ModifiedResNet, ViT:xxx")
    parser.add_argument('--txt_backbone', type=str, default="bert-base-uncased")
    parser.add_argument('--img_dim', type=int, default=768, help='dimension of image embedding vectors')
    parser.add_argument('--text_dim', type=int, default=768, help='dimension of text embedding vectors')
    parser.add_argument('--patch_size', type=int, default=16, help='Just for ViT model')
    parser.add_argument('--layers', type=list, default=[3, 4, 6, 3], help='Just for ModifiedResNet model')
    parser.add_argument('--heads', type=int, default=8, help='Just for ModifiedResNet model')

    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)

    # 设置超参数
    parser.add_argument('--num_epoches', type=int, default=30)
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
            text_feat = model.encode_text(caption,mask)
        qids.append(pid.view(-1))  # flatten
        qfeats.append(text_feat)
    qids = torch.cat(qids, 0)
    qfeats = torch.cat(qfeats, 0)

    # image
    for pid, img in image_test_loader:
        img = img.to(device)
        with torch.no_grad():
            img_feat = model.encode_image(img)
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
        transforms.Normalize(args.mean,
                             args.std)
    ])

    image_test_loader,text_test_loader = get_loader_test(args.image_dir, args.caption_path, transform,  args.batch_size,args.num_workers)

    ac_t2i_top1_best = 0.0
    ac_t2i_top5_best = 0.0
    ac_t2i_top10_best = 0.0
    mAP_best = 0.0
    best = 0
    dst_best = args.best_dir + "/model_best" + ".pth"
    model = Create_model(args).to(device)

    for i in range(0, args.num_epoches):
        if i%2!=0 and i!=args.num_epoches-1:
            continue
        model_file = os.path.join(args.model_path, str(i+1))+".pth.tar"
        print(model_file)
        if os.path.isdir(model_file):
            continue
        checkpoint = torch.load(model_file)
        model.image_encoder.load_state_dict(checkpoint["ImgEncoder_state_dict"])
        model.text_encoder.load_state_dict(checkpoint["TxtEncoder_state_dict"])
        ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, mAP = test(image_test_loader,text_test_loader, model)
        if ac_top1_t2i > ac_t2i_top1_best:
            ac_t2i_top1_best = ac_top1_t2i
            ac_t2i_top5_best = ac_top5_t2i
            ac_t2i_top10_best = ac_top10_t2i
            mAP_best = mAP
            best = i
            shutil.copyfile(model_file, dst_best)

    print('Epo{}:  {:.5f}  {:.5f}  {:.5f}  {:.5f}'.format(
            best, ac_t2i_top1_best, ac_t2i_top5_best, ac_t2i_top10_best, mAP_best))

if __name__ == '__main__':
    args = Test_parse_args()
    Test_main(args)