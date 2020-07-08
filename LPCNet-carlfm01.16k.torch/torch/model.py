import torch
import torch.nn as nn
import numpy as np
import math

class MDense(nn.Module):
    def __init__(self, input_features, output_features, channels=2, use_bias=True):
        super(MDense, self).__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.use_bias=use_bias

        self.weight1 = nn.Parameter(torch.Tensor(output_features, input_features).zero_())
        nn.init.xavier_uniform_(self.weight1 )
        self.weight2 = nn.Parameter(torch.Tensor(output_features, input_features).zero_())
        nn.init.xavier_uniform_(self.weight2)
        if use_bias:
            self.bias1 = nn.Parameter(torch.Tensor(output_features).zero_())
            self.bias2 = nn.Parameter(torch.Tensor(output_features).zero_())
        else:
            self.register_parameter('bias', None)
        self.factor1 = nn.Parameter(torch.ones(output_features))
        self.factor2 = nn.Parameter(torch.ones(output_features))

    def forward(self, inputs):
        output1 = inputs.matmul(self.weight1.t())
        output2 = inputs.matmul(self.weight2.t())
        if self.use_bias:
            output1 = output1 + self.bias1
            output2 = output2 + self.bias2
        output1 = torch.tanh(output1) * self.factor1
        output2 = torch.tanh(output2) * self.factor2
        output = output1 + output2
        #pytorch中不做softmax
        #output_final = torch.softmax(output, dim = 1)
        return output#output_final


class LPCNet(nn.Module):
    def __init__(self, hparams):
        super(LPCNet, self).__init__()
        self.embedding_size = hparams.embedding_size
        self.frame_size  = hparams.frame_size
        #48k采样率应该为256*3
        self.embed_pitch = nn.Embedding(hparams.pitch_max_period, hparams.embedding_pitch_size)
        self.embed_sig   = nn.Embedding(256, hparams.embedding_size)
        self.embed_exc   = nn.Embedding(256, hparams.embedding_size)

        num_rows = 256
        num_cols = hparams.embedding_size
        init_val = np.random.uniform(-1.7321, 1.7321, (num_rows, num_cols))
        init_val = init_val + np.reshape(math.sqrt(12) * np.arange(-.5 * num_rows + .5, .5 * num_rows - .4) / num_rows, (num_rows, 1))
        init_val *= 0.1

        self.embed_sig.weight.data.copy_(torch.Tensor(init_val))
        self.embed_exc.weight.data.copy_(torch.Tensor(init_val))


        self.feature_conv1 = nn.Conv1d(hparams.embedding_pitch_size + hparams.nb_used_features, 128, kernel_size=3, stride=1, padding=1)
        self.feature_conv2 = nn.Conv1d(128, hparams.embedding_pitch_size + hparams.nb_used_features, kernel_size=3, stride=1, padding=1)
        self.feature_dense1= nn.Linear(hparams.embedding_pitch_size + hparams.nb_used_features, 128)
        self.feature_dense2= nn.Linear(128, 128)
        self.tanh           = nn.Tanh()
        self.gru_a          = nn.GRU(512, hparams.rnn_units1, batch_first=True)
        self.gru_b          = nn.GRU(512, hparams.rnn_units2, batch_first=True)
        self.md             = MDense(16, 256)



    def forward(self, x):
        #in_data,in_exc,features,periods,out_exc
        #x0: [16,7200,2]
        #x1: [16,7200,1]
        #x2: [16,15,63]
        #x3: [16,15,1]

        #[sig, pred] ==> x[0]
        #[16,7200,2] ==> [16,7200,2, 128]
        cpcm = self.embed_sig(x[0].cuda())
        #[16,7200,2, 128]==>[16, 7200, 256]
        cpcm2 = cpcm.reshape(cpcm.size(0), cpcm.size(1), -1)
        #[16, 7200, 1] ==> [16,7200,1, 128]
        cexc = self.embed_exc(x[1].cuda())
        #[16,7200,1, 128] ==> [16, 7200, 128]
        cexc2 = cexc.reshape(cexc.size(0), -1, self.embedding_size)

        #[16,15,1] ==>[16, 15, 1, 64]
        pitch = self.embed_pitch(x[3].cuda())
        #[16, 15, 1, 64]==>[16, 15, 64]
        pitch2 = pitch.reshape(pitch.size(0), -1, 64)

        #[16, 15, 63] [16, 15, 64] ==> [16, 15, 127]
        cat_feat = torch.cat((x[2].cuda(), pitch2), 2)

        #这里注意维度。
        #[16, 15, 127] ==> [16, 127, 15]
        cat_feat1 = cat_feat.permute(0, 2, 1)
        #[16, 127, 15] ==> [16, 128, 15]
        c_feat2 = self.tanh(self.feature_conv1(cat_feat1))
        # [16, 128 ,15] ==> [16, 127 ,15]
        cfeat = self.tanh(self.feature_conv2(c_feat2))
        # [16, 127 ,15] ==> [16, 15 ,127]
        c_feat2 = cfeat.permute(0, 2, 1)

        cfeat_add = cat_feat + c_feat2

        #[16, 15, 127] ==> [16, 15, 128]
        fd1 = self.tanh(self.feature_dense1(cfeat_add))
        # [16, 15, 128] ==> [16, 15, 128]
        fd2 = self.tanh(self.feature_dense2(fd1))

        #实现repeat_elements功能
        #[16, 15, 128] ==> [16, 15, 1, 128]
        fd2_uns = fd2.unsqueeze(2)
        # [16, 15, 1, 128] ==> [16, 15, 480, 128]
        fd2_exp = fd2_uns.repeat(1, 1, self.frame_size, 1)
        #[16, 15, 480, 128] ==> [16, 7200, 128]
        fd2_final = fd2_exp.reshape(fd2_exp.shape[0], -1, fd2_exp.shape[-1])

        rnn_in = torch.cat((cpcm2, cexc2, fd2_final), 2)
        # [16, 7200, 256] [16, 7200, 128] [16, 7200, 128]==> [16, 7200, 512]
        gru_out1,ttt = self.gru_a(rnn_in)

        rnn_in2 = torch.cat((gru_out1, fd2_final), 2)
        gru_out2, ttb = self.gru_b(rnn_in2)
        ulaw_prob = self.md(gru_out2)

        return ulaw_prob

    def encoder(self, feat, pitch):
        # [1,483,1] ==>[1, 483, 1, 64]
        pitch1 = self.embed_pitch(torch.LongTensor(pitch).cuda())
        # [16, 15, 1, 64]==>[16, 15, 64]
        pitch2 = pitch1.reshape(pitch1.size(0), -1, 64)

        # [1, 483, 63] [1, 483, 64] ==> [1, 483, 127]
        cat_feat = torch.cat((torch.FloatTensor(feat).cuda(), pitch2), 2)

        # 这里注意维度。
        # [1, 483, 127] ==> [1, 127, 483]
        cat_feat1 = cat_feat.permute(0, 2, 1)
        # [1, 127, 483] ==> [16, 128, 483]
        c_feat2 = self.tanh(self.feature_conv1(cat_feat1))
        # [1, 128 ,483] ==> [1, 127 ,483]
        cfeat = self.tanh(self.feature_conv2(c_feat2))
        # [1, 127 ,483] ==> [1, 483 ,127]
        c_feat2 = cfeat.permute(0, 2, 1)

        cfeat_add = cat_feat + c_feat2

        # [1, 483, 127] ==> [1, 483, 128]
        fd1 = self.tanh(self.feature_dense1(cfeat_add))
        # [1, 483, 128] ==> [1, 483, 128]
        fd2 = self.tanh(self.feature_dense2(fd1))

        return fd2

    def decoder(self, pcm, exc, dec_feat, dec_state1, dec_state2):
        # [1,1,2] ==> [1,1,2, 128]
        cpcm = self.embed_sig(torch.LongTensor(pcm).cuda())
        # [1,1,2, 128]==>[1, 1, 256]
        cpcm2 = cpcm.reshape(cpcm.size(0), cpcm.size(1), -1)

        # [1, 1, 1] ==> [1,1,1, 128]
        cexc = self.embed_exc(torch.LongTensor(exc).cuda())
        # [1,1,1, 128] ==> [1, 1, 128]
        cexc2 = cexc.reshape(cexc.size(0), -1, self.embedding_size)

        # [1, 1, 256] [1, 1, 128] [1, 1, 128]==> [1, 1, 512]
        rnn_in = torch.cat((cpcm2, cexc2, dec_feat), 2)

        gru_out1, state1 = self.gru_a(rnn_in, dec_state1)

        rnn_in2 = torch.cat((gru_out1, dec_feat), 2)
        gru_out2, state2 = self.gru_b(rnn_in2, dec_state2)
        ulaw_prob = self.md(gru_out2)
        #pytorch中训练的时候不做softmax
        output_final = torch.softmax(ulaw_prob, dim = 2)
        return output_final, state1, state2
