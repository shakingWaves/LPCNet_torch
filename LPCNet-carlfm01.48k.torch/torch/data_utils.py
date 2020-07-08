'''
    Read feature file and mu-law file
'''
import torch
import numpy as np
import sys

#打开文件列表
def load_feature_and_pcm(filename):
    with open(filename, encoding='utf-8') as f:
        filepaths = [line.strip().split() for line in f]
        return filepaths

class FeaturePCMLoader(torch.utils.data.Dataset):
    def __init__(self, set_path, frame_size, nb_used_features, bfcc_band, nb_features, pitch_idx, batch_chunk):
        self.feature_and_pcm = load_feature_and_pcm(set_path)
        self.frame_size = frame_size
        self.nb_features = nb_features  # NB_FEATURES (2*NB_BANDS+3+LPC_ORDER)
        self.nb_used_features = nb_used_features
        self.bfcc_band = bfcc_band
        self.pitch_idx   = pitch_idx
        self.batch_chunk = batch_chunk  #作为len标记
        self.len = sys.maxsize               #设置一个非常大的值
        self.index = 0
        self.batch_x00 = []
        self.batch_x01 = []
        self.batch_x02 = []
        self.batch_x03 = []
        self.batch_y10 = []

    def process_feature_pcm(self, feature_file, pcm_file):

        frame_size = self.frame_size
        nb_features = self.nb_features  # NB_FEATURES (2*NB_BANDS+3+LPC_ORDER)
        bfcc_band = self.bfcc_band
        nb_used_features = self.nb_used_features
        feature_chunk_size = 15
        pcm_chunk_size = frame_size * feature_chunk_size

        # u for unquantised, load 16 bit PCM samples and convert to mu-law

        data = np.fromfile(pcm_file, dtype='uint8')
        nb_frames = len(data) // (4 * pcm_chunk_size)

        features = np.fromfile(feature_file, dtype='float32')

        # limit to discrete number of frames
        data = data[:nb_frames * 4 * pcm_chunk_size]
        features = features[:nb_frames * feature_chunk_size * nb_features]

        features = np.reshape(features, (nb_frames * feature_chunk_size, nb_features))

        sig = np.reshape(data[0::4], (nb_frames, pcm_chunk_size, 1))
        pred = np.reshape(data[1::4], (nb_frames, pcm_chunk_size, 1))
        in_exc = np.reshape(data[2::4], (nb_frames, pcm_chunk_size, 1))
        out_exc = np.reshape(data[3::4], (nb_frames, pcm_chunk_size, 1))
        del data

        #print("ulaw std = ", np.std(out_exc))

        features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))
        features = features[:, :, :nb_used_features]
        features[:, :, bfcc_band:bfcc_band*2] = 0

        periods = (.1 + 50 * features[:, :, self.pitch_idx:self.pitch_idx + 1] + 100).astype('int16')

        in_data = np.concatenate([sig, pred], axis=-1)

        del sig
        del pred
        return [in_data, in_exc, features, periods], out_exc

    def get_feature_pcm(self):
        while (len(self.batch_x00) < self.batch_chunk):
            if (self.index < len(self.feature_and_pcm)):
                feature_file, pcm_file =  self.feature_and_pcm[self.index]
                self.index = self.index + 1
            else:
                return [None, None, None, None], None
            [in_data, in_exc, features, periods], out_exc = self.process_feature_pcm(feature_file, pcm_file)

            self.batch_x00.extend(in_data)
            self.batch_x01.extend(in_exc)
            self.batch_x02.extend(features)
            self.batch_x03.extend(periods)
            self.batch_y10.extend(out_exc)


        x_0_0 = self.batch_x00[:self.batch_chunk]
        self.batch_x00 = self.batch_x00[self.batch_chunk:]
        x_0_1 = self.batch_x01[:self.batch_chunk]
        self.batch_x01 = self.batch_x01[self.batch_chunk:]
        x_0_2 = self.batch_x02[:self.batch_chunk]
        self.batch_x02 = self.batch_x02[self.batch_chunk:]
        x_0_3 = self.batch_x03[:self.batch_chunk]
        self.batch_x03 = self.batch_x03[self.batch_chunk:]
        y_1_0 = self.batch_y10[:self.batch_chunk]
        self.batch_y10 = self.batch_y10[self.batch_chunk:]

        return [torch.LongTensor(x_0_0), torch.LongTensor(x_0_1), torch.FloatTensor(x_0_2), torch.LongTensor(x_0_3)], torch.LongTensor(y_1_0)


    def __getitem__(self, index):
        return self.get_feature_pcm()

    def __len__(self):
        return self.len

class FeaturePCMCollate():
    def __call__(self, batch):
        return batch[0][0], batch[0][1]
