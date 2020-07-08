'''
    LPCNet pytorch版本
    根据模型和输入特征预测
'''
import argparse
from model import LPCNet
from hparams import create_hparams
import numpy as np
import torch
import os
from ulaw import ulaw2lin, lin2ulaw
import torch
#from tools.keras_to_pytorch import model_init

def synthesis(args, hparams):
    model = LPCNet(hparams).cuda()
    feature_file = args.feature_file;
    out_file = args.out_file

    frame_size = hparams.frame_size
    nb_features = hparams.nb_features

    features = np.fromfile(feature_file, dtype='float32')
    features = features.reshape(-1, nb_features)
    #features = np.resize(features, (-1, nb_features)) #使用resize会导致最后一行数据丢失

    nb_frames = 1

    feature_chunk_size = features.shape[0]
    pcm_chunk_size = frame_size * feature_chunk_size
    features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))
    periods = (.1 + 50*features[:,:,hparams.pitch_idx:hparams.pitch_idx+1]+100).astype('int16')

    if None == hparams.checkpoint_file or not os.path.isfile(hparams.checkpoint_file):
        return
    checkpoint_dict = torch.load(hparams.checkpoint_file)
    model.load_state_dict(checkpoint_dict['state_dict'])

    #model_init(model)

    model.eval()

    order = 16

    pcm = np.zeros((nb_frames * pcm_chunk_size,))
    fexc = np.zeros((1, 1, 2), dtype='float32')
    iexc = np.zeros((1, 1, 1), dtype='int16')
    state1 = torch.Tensor(np.zeros((1, 1, hparams.rnn_units1), dtype='float32')).cuda()
    state2 = torch.Tensor(np.zeros((1, 1, hparams.rnn_units2), dtype='float32')).cuda()

    mem = 0
    coef = 0.85

    fout = open(out_file, "wb")
    skip = order + 1

    for c in range(0, nb_frames):
        cfeat = model.encoder(features[c:c+1, :, :nb_features], periods[c:c+1, :, :])
        fexc[0, 0, 0] = 128  # 0 mulaw
        iexc[0, 0, 0] = 128
        for fr in range(0, feature_chunk_size):
            f = c * feature_chunk_size + fr
            a = features[c, fr, nb_features - order:]
            for i in range(skip, frame_size):
                pred = -sum(a*pcm[f*frame_size + i - 1:f*frame_size + i - order-1:-1])
                fexc[0, 0, 1] = lin2ulaw(pred)

                p_tensor, state1, state2 = model.decoder(fexc, iexc, cfeat[:, fr:fr+1, :], state1, state2)
                p = p_tensor.clone().cpu().detach().numpy()
                # Lower the temperature for voiced frames to reduce noisiness
                p *= np.power(p, np.maximum(0, 1.5 * features[c, fr, hparams.pitch_idx+1] - .5))
                p = p / (1e-18 + np.sum(p))
                # Cut off the tail of the remaining distribution
                p = np.maximum(p - 0.002, 0).astype('float64')
                p = p / (1e-8 + np.sum(p))

                iexc[0, 0, 0] = np.argmax(np.random.multinomial(1, p[0, 0, :], 1))
                pcm[f * frame_size + i] = pred + ulaw2lin(iexc[0, 0, 0])
                fexc[0, 0, 0] = lin2ulaw(pcm[f * frame_size + i])
                mem = coef * mem + pcm[f * frame_size + i]
                # print(mem)
                np.array([np.round(mem)], dtype='int16').tofile(fout)
            skip = 0

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('-f', '--feature_file', type=str,
                       default='E:/Research/Synthesis/BZNSYP/ttt/wxl/feature-9.f32', help='features file to train')
    parse.add_argument('-o', '--out_file', type=str,
                       default='E:/Research/Synthesis/BZNSYP/ttt/wxl/feature-9.s16', help='features file to train')
    args = parse.parse_args()
    hparams = create_hparams()
    synthesis(args, hparams)

if __name__=="__main__":
    main()