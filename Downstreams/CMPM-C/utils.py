import errno
import sys
import os.path as osp
import torch.utils.data as data
import os
import torch
import numpy as np
import random
import json
import torch.nn as nn
from einops.layers.torch import Rearrange
from transformers import get_linear_schedule_with_warmup
import yaml

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("The seed is {}".format(seed))

def lr_scheduler(optimizer, args,loader_length):
    if args.lr_decay_type == "get_linear_schedule_with_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(args.epochs * loader_length * 0.1/args.accumulation_steps),
            num_training_steps=int(args.epochs * loader_length/args.accumulation_steps)
        )
        print("lr_scheduler is get_linear_schedule_with_warmup")
    else:
        if '_' in args.epoches_decay:
            epoches_list = args.epoches_decay.split('_')
            epoches_list = [int(e) for e in epoches_list]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, epoches_list, gamma=args.lr_decay_ratio)
            print("lr_scheduler is MultiStepLR")
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(args.epoches_decay), gamma=args.lr_decay_ratio)
            print("lr_scheduler is StepLR")
    return scheduler

def save_checkpoint(state, epoch, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    filename = os.path.join(dst, str(epoch)) + '.pth.tar'
    torch.save(state, filename)

def gradual_warmup(epoch,optimizer,epochs):
    if epoch == 0:
        warmup_percent_done = (epoch + 1) / epochs
    else:
        warmup_percent_done = (epoch + 1) / epoch
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*warmup_percent_done
    return optimizer

def PrintInformation(args,model):
    # with open(args.annotation_path,"r") as f:
    #     dataset = json.load(f)
    #     num = len(dataset)
    print("The image model is: {}".format(args.img_backbone))
    print("The language model is: {}".format(args.txt_backbone))
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    print("Checkpoints are saved in: {}".format(args.name))
    print("The original learning rate is {}".format(args.lr))


def train_information_setting(args): #配置好训练信息的输出文件
    name = args.name
    # set some paths
    log_dir = args.log_dir
    log_dir = os.path.join(log_dir, name)
    sys.stdout = Logger(os.path.join(log_dir, "train_log.txt"))
    opt_dir = os.path.join('log', name)
    if not os.path.exists(opt_dir):
        os.makedirs(opt_dir)
    with open('%s/opts_train.yaml' % opt_dir, 'w') as fp:
        yaml.dump(vars(args), fp, default_flow_style=False)

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def compute_topk(query, gallery, target_query, target_gallery, k=[1,10], reverse=False):
    result = []
    query = query / (query.norm(dim=1,keepdim=True)+1e-12)
    gallery = gallery / (gallery.norm(dim=1,keepdim=True)+1e-12)
    sim_cosine = torch.matmul(query, gallery.t())
    result.extend(topk(sim_cosine, target_gallery, target_query, k))
    if reverse:
        result.extend(topk(sim_cosine, target_query, target_gallery, k, dim=0))
    return result

def topk(sim, target_gallery, target_query, k=[1,10], dim=1):
    result = []
    maxk = max(k)
    size_total = len(target_gallery)
    _, pred_index = sim.topk(maxk, dim, True, True)
    pred_labels = target_gallery[pred_index]
    if dim == 1:
        pred_labels = pred_labels.t()
    correct = pred_labels.eq(target_query.view(1,-1).expand_as(pred_labels))

    for topk in k:
        correct_k = torch.sum(correct[:topk], dim=0)
        correct_k = torch.sum(correct_k > 0).float()
        result.append(correct_k * 100 / size_total)
    return result

def test_map(query_feature,query_label,gallery_feature, gallery_label):
    query_feature = query_feature / (query_feature.norm(dim=1, keepdim=True) + 1e-12)
    gallery_feature = gallery_feature / (gallery_feature.norm(dim=1, keepdim=True) + 1e-12)
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i],  gallery_feature, gallery_label)

        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    CMC = CMC.float()
    CMC = CMC / len(query_label)
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))
    return CMC[0], CMC[4], CMC[9], ap / len(query_label)

def evaluate(qf, ql, gf, gl):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    index = np.argsort(score)
    index = index[::-1]
    gl=gl.cuda().data.cpu().numpy()
    ql=ql.cuda().data.cpu().numpy()
    query_index = np.argwhere(gl == ql)
    CMC_tmp = compute_mAP(index, query_index)
    return CMC_tmp

def compute_mAP(index, good_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc
    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc