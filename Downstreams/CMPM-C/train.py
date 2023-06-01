import argparse
from dataloader import get_loader
from model import Create_model
from loss import Loss_calc
from torchvision import transforms
from utils import *
import time
from optimizer import create_optimizer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def Train_parse_args():
    parser = argparse.ArgumentParser(description='command for train by Resnet')
    parser.add_argument('--mean', default=[0.357, 0.323, 0.328], type=list)
    parser.add_argument('--std', default=[0.252, 0.242, 0.239], type=list)
#***********************************************************************************************************************
    #设置数据集路径和输出路径等
    parser.add_argument('--name', default='CUHK-CMPMC', type=str, help='output model name')
    parser.add_argument('--checkpoint_dir', type=str,
                        default="log",
                        help='directory to store checkpoint')
    parser.add_argument('--log_dir', type=str,
                        default="log",
                        help='directory to store log')
    parser.add_argument('--image_path', type=str,
                        default=r'data/CUHK-PEDES',
                        help='directory to store dataset')
    parser.add_argument('--dataset_path', type=str,
                        default=r'data/CUHK-PEDES-train-depart.json',
                        help='directory to annotations')
    parser.add_argument('--checkpoint_path', type=str,default=r'checkpoints/PLIP_RN50.pth.tar')
#***********************************************************************************************************************
    #设置模型backbone的类型和参数
    parser.add_argument('--img_backbone', type=str, default='ModifiedResNet',help="ResNet:xxx, ModifiedResNet, ViT:xxx")
    parser.add_argument('--txt_backbone', type=str, default="bert-base-uncased")
    parser.add_argument('--img_dim', type=int, default=768, help='dimension of image embedding vectors')
    parser.add_argument('--text_dim', type=int, default=768, help='dimension of text embedding vectors')
    parser.add_argument('--layers', type=list, default=[3,4,6,3], help='Just for ModifiedResNet model')
    parser.add_argument('--heads', type=int, default=8, help='Just for ModifiedResNet model')
    parser.add_argument('--feature_size', type=int, default=768)
    parser.add_argument('--num_classes', type=int, default=11003)  # CUHK:11003 ICFG:3102 THE NUMBER OF IDENTITIES
#***********************************************************************************************************************
    #设置训练预处理超参数
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)

#***********************************************************************************************************************
    #设置学习率等超参数
    parser.add_argument('--save_every', type=int, default=1, help='step size for saving trained models')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--warm_epoch', default=5, type=int, help='the first K epoch that needs warm up')
    parser.add_argument('--lr', type=float, default=0.0001, help='the learning rate of optimizer')
    parser.add_argument('--lr_decay_type', type=str, default='MultiStepLR',
                        help='One of "MultiStepLR" or "StepLR" or "get_linear_schedule_with_warmup"')
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
    parser.add_argument('--epoches_decay', type=str, default='20', help='#epoches when learning rate decays')
    parser.add_argument('--backbone_frozen', type=bool, default=False)
#***********************************************************************************************************************
    # 设置优化器超参数
    parser.add_argument('--optimizer', type=str, default="adan", help='The optimizer type:adam or adan')

    parser.add_argument('--adam_alpha', type=float, default=0.9)
    parser.add_argument('--adam_beta', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-8)

    parser.add_argument('--adan_max-grad-norm', type=float, default=0.0,
                        help='if the l2 norm is large than this hyper-parameter, then we clip the gradient  (default: 0.0, no gradient clip)')
    parser.add_argument('--adan_weight-decay', type=float, default=0.02,
                        help='weight decay, similar one used in AdamW (default: 0.02)')
    parser.add_argument('--adan_opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='optimizer epsilon to avoid the bad case where second-order moment is zero (default: None, use opt default 1e-8 in adan)')
    parser.add_argument('--adan_opt-betas', default=[0.98, 0.92, 0.99], type=float, nargs='+', metavar='BETA',
                        help='optimizer betas in Adan (default: None, use opt default [0.98, 0.92, 0.99] in Adan)')
    parser.add_argument('--adan_no-prox', action='store_true', default=False,
                        help='whether perform weight decay like AdamW (default=False)')

#***********************************************************************************************************************
    #其他设置
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpus', type=str, default="0,1,2,3")
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()
    return args


def train(args):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")

    train_information_setting(args)
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.name)

    transform = transforms.Compose([
        transforms.Resize((args.height, args.width),interpolation=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(args.mean,args.std)
    ])

    epochs = args.epochs

    model = Create_model(args).to(device)
    model_file = args.checkpoint_path
    checkpoint = torch.load(model_file)
    model.image_encoder.load_state_dict(checkpoint["ImgEncoder_state_dict"])
    model.text_encoder.load_state_dict(checkpoint["TxtEncoder_state_dict"])
    Loss = Loss_calc(args).to(device)
    params = [{"params": model.image_encoder.parameters(), "lr": args.lr},
                  {"params": model.text_encoder.parameters(), "lr": args.lr},
                  {"params": Loss.parameters(), "lr": args.lr * 10}]


    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # 多卡bn层同步
    optimizer = create_optimizer(params, args)

    train_dataloader = get_loader(args.image_path, args.dataset_path, transform,  None,args.batch_size,args.num_workers,distributed=True)
    scheduler = lr_scheduler(optimizer, args,len(train_dataloader))
    PrintInformation(args, model)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=True,find_unused_parameters=True)

    if rank == 0:
        print("            =======  Training  ======= \n")

    model.train()

    for epoch in range(epochs):
        print("**********************************************************")
        print(f">>> Training epoch {epoch + 1}")
        total_loss = 0
        start = time.time()
        train_dataloader.sampler.set_epoch(epoch)
        if epoch < args.warm_epoch:
            print('learning rate warm_up')
            optimizer = gradual_warmup(epoch, optimizer, epochs=args.warm_epoch)

        optimizer.zero_grad()
        for idx, (images_gt, targets,  masks,  labels) in enumerate(
                train_dataloader):
            images_gt,  targets,  masks,  labels = images_gt.to(device),  targets.to(device),  masks.to(device),  labels.to(device) - 1
            global_visual_embed,  global_textual_embed = model(images_gt, targets, masks)
            IDlabels = labels
            cmpm_loss, cmpc_loss, loss, image_precision, text_precision = Loss(global_visual_embed, global_textual_embed, IDlabels)
            #print(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if idx % 50 == 0 :
                print(
                    "Train Epoch:[{}/{}] iteration:[{}/{}] cmpm_loss:{:.4f} cmpc_loss:{:.4f} "
                "image_pre:{:.4f} text_pre:{:.4f}"
                    .format(epoch + 1, args.epochs, idx, len(train_dataloader), cmpm_loss.item(),
                            cmpc_loss.item(),
                            image_precision, text_precision))
        scheduler.step()
        Epoch_time = time.time() - start
        print("Average loss is :{}".format(total_loss / len(train_dataloader)))
        print('Epoch_training complete in {:.0f}m {:.0f}s'.format(
            Epoch_time // 60, Epoch_time % 60))

        if epoch % args.save_every == 0 or epoch == epochs - 1:
            state = {"epoch": epoch + 1,
                     "ImgEncoder_state_dict": model.module.image_encoder.state_dict(),
                     "TxtEncoder_state_dict": model.module.text_encoder.state_dict()
                     }
            save_checkpoint(state, epoch + 1, checkpoint_dir)

def main():
    args = Train_parse_args()
    train(args)

if __name__ == '__main__':
    main()
