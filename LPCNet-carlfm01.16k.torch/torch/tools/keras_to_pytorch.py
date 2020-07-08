import numpy as np
import torch
from hparams import create_hparams
from model import LPCNet

#该函数主要用于使用keras模型的参数初始化模型
def convert_input_kernel(kernel):
    kernel_z, kernel_r, kernel_h = np.hsplit(kernel, 3)
    kernels = [kernel_r, kernel_z, kernel_h]
    return np.vstack([k.reshape(k.T.shape) for k in kernels])

def convert_recurrent_kernel(kernel):
    kernel_z, kernel_r, kernel_h = np.hsplit(kernel, 3)
    kernels = [kernel_r, kernel_z, kernel_h]
    return np.vstack(kernels)

def convert_bias(bias):
    bias = bias.reshape(2, 3, -1)
    bias = bias[:, [1, 0, 2], :]
    bias = bias.reshape(-1)
    return bias

#pytorch to keras
def pk_convert_input_kernel(kernel):
    kernel_r, kernel_z, kernel_h = np.vsplit(kernel, 3)
    kernels = [kernel_z.T, kernel_r.T, kernel_h.T]
    return np.hstack(kernels)

def pk_convert_recurrent_kernel(kernel):
    kernel_r, kernel_z, kernel_h = np.vsplit(kernel, 3)
    kernels = [kernel_z.T, kernel_r.T, kernel_h.T]
    return np.hstack(kernels)

def pk_convert_bias(bias):
    bias = bias.reshape(2, 3, -1)
    return bias[:, [1, 0, 2], :].reshape(-1)


def model_init(model):

    embed_exc = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/embed_exc_weight.txt")
    model.embed_exc.weight.data.copy_(torch.FloatTensor(embed_exc))

    embed_pitch = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/embed_pitch_weight.txt")
    model.embed_pitch.weight.data.copy_(torch.FloatTensor(embed_pitch))

    embed_sig = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/embed_sig_weight.txt")
    model.embed_sig.weight.data.copy_(torch.FloatTensor(embed_sig))

    feature_conv1_weight = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/feature_conv1_weight.txt")
    feature_conv1_weight = feature_conv1_weight.reshape(3, 127, 128)
    feature_conv1_weight = feature_conv1_weight.transpose(2, 1, 0);
    model.feature_conv1.weight.data.copy_(torch.Tensor(feature_conv1_weight))

    feature_conv1_bias = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/feature_conv1_bias.txt")
    model.feature_conv1.bias.data.copy_(torch.Tensor(feature_conv1_bias))

    #feature conv2
    feature_conv2_weight = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/feature_conv2_weight.txt")
    feature_conv2_weight = feature_conv2_weight.reshape(3, 128, 127)
    feature_conv2_weight = feature_conv2_weight.transpose(2, 1, 0);
    model.feature_conv2.weight.data.copy_(torch.Tensor(feature_conv2_weight))

    feature_conv2_bias = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/feature_conv2_bias.txt")
    model.feature_conv2.bias.data.copy_(torch.Tensor(feature_conv2_bias))

    #feature dense1
    feature_dense1_weight = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/feature_dense1_weight.txt")
    feature_dense1_weight = feature_dense1_weight.transpose(1, 0);
    model.feature_dense1.weight.data.copy_(torch.Tensor(feature_dense1_weight))
    feature_dense1_bias = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/feature_dense1_bias.txt")
    model.feature_dense1.bias.data.copy_(torch.Tensor(feature_dense1_bias))

    #feature dense2
    feature_dense2_weight = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/feature_dense2_weight.txt")
    feature_dense2_weight = feature_dense2_weight.transpose(1, 0);
    model.feature_dense2.weight.data.copy_(torch.Tensor(feature_dense2_weight))
    feature_dense2_bias = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/feature_dense2_bias.txt")
    model.feature_dense2.bias.data.copy_(torch.Tensor(feature_dense2_bias))

    #gru_a
    gru_a_hh_weight = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/gru_a_hh_weight.txt")
    gru_a_ih_weight = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/gru_a_ih_weight.txt")
    gru_a_bias   = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/gru_a_bias.txt")

    a_weight_ih = torch.from_numpy(convert_input_kernel(gru_a_ih_weight))
    a_weight_hh = torch.from_numpy(convert_recurrent_kernel(gru_a_hh_weight))
    a_weight_bias = torch.from_numpy(convert_bias(gru_a_bias))

    model.gru_a.weight_ih_l0.data.copy_(a_weight_ih)
    model.gru_a.weight_hh_l0.data.copy_(a_weight_hh)
    model.gru_a.bias_ih_l0.data.copy_(a_weight_bias[:1152])
    model.gru_a.bias_hh_l0.data.copy_(a_weight_bias[1152:])

    #gru_b
    gru_b_hh_weight = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/gru_b_hh_weight.txt")
    gru_b_ih_weight = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/gru_b_ih_weight.txt")
    gru_b_bias = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/gru_b_bias.txt")

    b_weight_ih = torch.from_numpy(convert_input_kernel(gru_b_ih_weight))
    b_weight_hh = torch.from_numpy(convert_recurrent_kernel(gru_b_hh_weight))
    b_weight_bias = torch.from_numpy(convert_bias(gru_b_bias))

    model.gru_b.weight_ih_l0.data.copy_(b_weight_ih)
    model.gru_b.weight_hh_l0.data.copy_(b_weight_hh)
    model.gru_b.bias_ih_l0.data.copy_(b_weight_bias[:48])
    model.gru_b.bias_hh_l0.data.copy_(b_weight_bias[48:])


    #md
    dual_fc_weight1 = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/dual_fc_w0_weight.txt")
    dual_fc_weight2 = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/dual_fc_w1_weight.txt")
    dual_fc_bias1   = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/dual_fc_b0_weight.txt")
    dual_fc_bias2   = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/dual_fc_b1_weight.txt")
    dual_fc_factor1 = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/dual_fc_f0_weight.txt")
    dual_fc_factor2 = np.loadtxt("E:/Research/Synthesis/LPCNet-carlfm01.48k/data/output/dual_fc_f1_weight.txt")
    model.md.weight1.data.copy_(torch.Tensor(dual_fc_weight1))
    model.md.weight2.data.copy_(torch.Tensor(dual_fc_weight2))
    model.md.bias1.data.copy_(torch.Tensor(dual_fc_bias1))
    model.md.bias2.data.copy_(torch.Tensor(dual_fc_bias2))
    model.md.factor1.data.copy_(torch.Tensor(dual_fc_factor1))
    model.md.factor2.data.copy_(torch.Tensor(dual_fc_factor2))


if __name__=='__main__':
    hparams = create_hparams()
    model = LPCNet(hparams)
    model_init(model)