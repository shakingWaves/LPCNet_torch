'''
    LPCNet pytorch版本
'''
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import sys
import argparse
from hparams import create_hparams
import torch.nn.functional as F
from data_utils import FeaturePCMLoader, FeaturePCMCollate
from logger import LPCNetLogger
from model import LPCNet
import os
import time

def prepare_logger(log_dir):
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logger = LPCNetLogger(log_dir)
    return logger

def prepare_dataloaders(hparams):
    train_file = hparams.training_files
    val_file   = hparams.validation_files
    test_file  = hparams.test_files
    collate_fn = FeaturePCMCollate()
    train_set = FeaturePCMLoader(train_file, hparams.frame_size, hparams.nb_used_features, hparams.bfcc_band, hparams.nb_features, hparams.pitch_idx ,hparams.batch_chunk)
    val_set = FeaturePCMLoader(val_file, hparams.frame_size, hparams.nb_used_features, hparams.bfcc_band, hparams.nb_features, hparams.pitch_idx, hparams.batch_chunk)
    test_set = FeaturePCMLoader(test_file, hparams.frame_size, hparams.nb_used_features, hparams.bfcc_band, hparams.nb_features, hparams.pitch_idx, hparams.batch_chunk)


    #音频处理的时候是按照chunk处理，我们在读音频的时候是一个个读取的
    #所以，一个音频里可能有多个chunk，所以，batch_size设置为1,使用
    #collate_fn按照chunk处理
    train_loader = DataLoader(train_set,
        num_workers = 1, shuffle = hparams.shuffle,
        batch_size = hparams.batch_size, pin_memory = False,
        drop_last = True, collate_fn = collate_fn
    )

    #验证和测试集都按照一个音频作为处理单位
    val_loader = DataLoader(val_set,
                              num_workers=1, shuffle=hparams.shuffle,
                              batch_size=1, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn
                              )

    test_loader = DataLoader(test_set,
                            num_workers=1, shuffle=hparams.shuffle,
                            batch_size=1, pin_memory=False,
                            drop_last=True, collate_fn=collate_fn
                            )


    return train_loader, val_loader, test_loader


def save_checkpoint(model, optimizer, learning_rate, epoc, checkpoint_path):
    print("Saving model and optimizer state at epoc {} to {}".format(
        epoc, checkpoint_path))
    torch.save(
        {'epoc':epoc,
         'state_dict': model.state_dict(),
         'optimizer':optimizer.state_dict(),
         'learning_rate':learning_rate},
        checkpoint_path
    )

def load_checkpoint(checkpoint_file, model, optimizer):
    checkpoint_dict = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    epoc = checkpoint_dict['epoc']
    learning_rate = checkpoint_dict["learning_rate"]
    #设置学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    print("Loaded checkpoint '{}' from epoc {}".format(
        checkpoint_file, epoc))
    return model, optimizer ,epoc


def validate(model, val_loader, criterion, iteration ,logger):
    model.eval()

    val_loss = 0;
    for i, (x, y) in enumerate(val_loader):
        if None == y:
            break
        y_pred = model(x)
        y_pred = y_pred.permute(0, 2, 1).cuda()
        y = y.squeeze(2).cuda()
        loss = criterion(y_pred, y)
        val_loss += loss.item()
    val_loss = val_loss / (i + 1)
    model.train()
    logger.log_validation(val_loss, iteration)

def sparse_gru_a(model, final_density, iteration, t_start, t_end):
    #keras里get_weights返回[[W_z; W_r; W_h], [U_z; U_r; U_h], [bias_z; bias_r; bias_h]]
    #W_z, W_r, W_h are weights converting input to hidden states
    #U_z, U_r, U_h are weights converting state to state
    #b_z, b_r, b_h are biases

    #将权重拷贝到新的内存中
    p = model.gru_a.weight_hh_l0.clone().detach().cpu().numpy()
    #p.shape=[1152, 384],keras是[384, 1152]，所以
    nb = p.shape[0] // p.shape[1]
    N = p.shape[1]
    for k in range(nb):
        density = final_density[k]
        if iteration < t_end:
            r = 1 - (iteration - t_start)/(t_end - t_start)
            density = 1 - (1-final_density[k]) * (1 - r*r*r)
        A = p[k*N:(k+1)*N, : ]
        #两个np.diag获取矩阵对角元素，其他位置为0
        A = A - np.diag(np.diag(A))
        #在keras里做了转置，这里不需要了，因为本来矩阵和keras就是转置的
        L = np.reshape(A, (N, N//16, 16))
        S = np.sum(L*L, axis=-1)
        #展平为一行,然后进行排序
        SS = np.sort(np.reshape(S, (-1, )))
        thresh = SS[round(N * N//16 * (1-density))]
        mask = (S >= thresh).astype('float32')
        mask = np.repeat(mask, 16, axis=1)
        mask = np.minimum(1, mask + np.diag(np.ones((N,))))
        p[k * N:(k + 1) * N, :] = p[k*N:(k+1)*N, : ] * mask
    #这里的copy_函数和keras的set_weights函数功能一样
    model.gru_a.weight_hh_l0.data.copy_(torch.Tensor(p))


def train(args, hparams):

    #模型
    #torch.set_default_dtype(torch.float64)
    model = LPCNet(hparams).cuda()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate, amsgrad=True)

    epoc_offset = 0
    if None != hparams.checkpoint_file and os.path.isfile(hparams.checkpoint_file):
        model, optimizer, epoc_offset = load_checkpoint(hparams.checkpoint_file, model, optimizer)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=(1- (hparams.lr_decay)))
    criteon   = nn.CrossEntropyLoss().cuda()

    #logger
    logger = prepare_logger(hparams.log_dir)

    train_loader, val_loader, test_loader = prepare_dataloaders(hparams)

    iteration = 0;
    tot_loss = 0.0
    for epoc in range(epoc_offset + 1, hparams.epochs):
        for i, (data,target) in enumerate(train_loader):
            if None == target:#如果为空，说明数据已经读取完
                break

            start = time.perf_counter()

            y_pred = model(data)
            #将class放到中间
            y_pred = y_pred.permute(0, 2, 1).cuda()
            target = target.squeeze(2).cuda()
            loss = criteon(y_pred, target)
            optimizer.zero_grad()
            loss.backward()

            #grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
            optimizer.step()
            scheduler.step()
            tot_loss = tot_loss + loss.item()
            avg_loss = tot_loss / (iteration+1)
            #print("epoc :", epoc, " i : ", i, " loss : ", loss.item())
            duration = time.perf_counter() - start
            if iteration % 10 == 0:
                logger.log_training("Train", loss.item(), avg_loss, optimizer.param_groups[0]["lr"], duration, iteration)
                print("epoc :", epoc, " i : ", i, " loss : ", loss.item(), "\tavg_loss : ", avg_loss)
            if iteration >= 1000 and iteration % 1000 == 0:
                validate(model, val_loader, criteon, iteration, logger)
            iteration += 1

            #这里对gru_a进行矩阵稀疏化,对应keras callback中的on_batch_end
            t_start = 2000
            t_end   = 40000
            interval = 400
            density = (0.05, 0.05, 0.2)
            #if iteration >= t_start and iteration <= t_end and (iteration - t_start) % interval == 0:
            #    sparse_gru_a(model, density, iteration, t_start, t_end)

        checkpoint_path = os.path.join(hparams.checkpoint_path, "pytorch_lpcnet20_384_10_G16_{:02d}.h5".format(epoc))
        save_checkpoint(model, optimizer, optimizer.param_groups[0]["lr"], epoc, checkpoint_path)

def main():

    parser = argparse.ArgumentParser()
    #parser.add_argument('-f', '--feature_file', type=str,
    #                   default='E:/Research/Synthesis/BZNSYP/features.16k.f32', help='features file to train')
    #parser.add_argument('-p', '--pcm_file', type=str,
    #                   default='E:/Research/Synthesis/BZNSYP/data.16k.u8', help='target pcm file')
    #parser.add_argument('--hparams', type=str,
    #                    required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams()

    train(args, hparams)


if __name__=='__main__':

    main()

