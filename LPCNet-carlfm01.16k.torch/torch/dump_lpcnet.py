from model import LPCNet, MDense
from hparams import create_hparams
import torch
import os
import numpy as np
from tools.keras_to_pytorch import model_init
import torch.nn as nn


max_rnn_neurons = 1
max_conv_inputs = 1
max_mdense_tmp = 1

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

def dump_layer_ignore(self, name, f, hf):
    print("ignoring layer " + name + " of type " + self.__class__.__name__)
    return False
nn.Module.dump_layer = dump_layer_ignore


def printSparseVector(f, A, name):
    N = A.shape[0]
    W = np.zeros((0,))
    diag = np.concatenate([np.diag(A[:,:N]), np.diag(A[:,N:2*N]), np.diag(A[:,2*N:])])
    A[:,:N] = A[:,:N] - np.diag(np.diag(A[:,:N]))
    A[:,N:2*N] = A[:,N:2*N] - np.diag(np.diag(A[:,N:2*N]))
    A[:,2*N:] = A[:,2*N:] - np.diag(np.diag(A[:,2*N:]))
    printVector(f, diag, name + '_diag')
    idx = np.zeros((0,), dtype='int')
    for i in range(3*N//16):
        pos = idx.shape[0]
        idx = np.append(idx, -1)
        nb_nonzero = 0
        for j in range(N):
            if np.sum(np.abs(A[j, i*16:(i+1)*16])) > 1e-10:
                nb_nonzero = nb_nonzero + 1
                idx = np.append(idx, j)
                W = np.concatenate([W, A[j, i*16:(i+1)*16]])
        idx[pos] = nb_nonzero
    printVector(f, W, name)
    #idx = np.tile(np.concatenate([np.array([N]), np.arange(N)]), 3*N//16)
    printVector(f, idx, name + '_idx', dtype='int')
    return;

def printVector(f, vector, name, dtype='float'):
    v = np.reshape(vector, (-1));
    #print('static const float ', name, '[', len(v), '] = \n', file=f)
    f.write('static const {} {}[{}] = {{\n   '.format(dtype, name, len(v)))
    for i in range(0, len(v)):
        f.write('{}'.format(v[i]))
        if (i!=len(v)-1):
            f.write(',')
        else:
            break;
        if (i%8==7):
            f.write("\n   ")
        else:
            f.write(" ")
    #print(v, file=f)
    f.write('\n};\n\n')
    return;

def dump_embedding_layer_impl(name, weights, f, hf):
    printVector(f, weights, name + '_weights')
    f.write('const EmbeddingLayer {} = {{\n   {}_weights,\n   {}, {}\n}};\n\n'
            .format(name, name, weights.shape[0], weights.shape[1]))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights.shape[1]))
    hf.write('extern const EmbeddingLayer {};\n\n'.format(name));

def dump_dense_layer_impl(name, weights, bias, activation, f, hf):
    printVector(f, weights, name + '_weights')
    printVector(f, bias, name + '_bias')
    f.write('const DenseLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}, {}, ACTIVATION_{}\n}};\n\n'
            .format(name, name, name, weights.shape[0], weights.shape[1], activation))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights.shape[1]))
    hf.write('extern const DenseLayer {};\n\n'.format(name));


def dump_sparse_gru(self, f, hf):
    global max_rnn_neurons
    name = 'sparse_gru_a'
    print("printing layer " + name + " of type sparse " + self.__class__.__name__)
    weight_ih_l0 = pk_convert_input_kernel(self.weight_ih_l0.detach().numpy())
    weight_hh_l0 = pk_convert_recurrent_kernel(self.weight_hh_l0.detach().numpy())

    bias_ih_l0 = self.bias_ih_l0.detach().numpy().reshape(-1)
    bias_hh_l0 = self.bias_hh_l0.detach().numpy().reshape(-1)
    bias = np.concatenate((bias_ih_l0, bias_hh_l0))
    bias = pk_convert_bias(bias)

    printSparseVector(f, weight_hh_l0, name + '_recurrent_weights')
    printVector(f, bias, name + '_bias')
    if hasattr(self, 'activation'):
        activation = self.activation.__name__.upper()
    else:
        activation = 'TANH'
    if hasattr(self, 'reset_after') and not self.reset_after:
        reset_after = 0
    else:
        reset_after = 1
    neurons = weight_ih_l0.shape[1]//3
    max_rnn_neurons = max(max_rnn_neurons, neurons)
    f.write('const SparseGRULayer {} = {{\n   {}_bias,\n   {}_recurrent_weights_diag,\n   {}_recurrent_weights,\n   {}_recurrent_weights_idx,\n   {}, ACTIVATION_{}, {}\n}};\n\n'
            .format(name, name, name, name, name, weight_ih_l0.shape[1]//3, activation, reset_after))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weight_ih_l0.shape[1]//3))
    hf.write('#define {}_STATE_SIZE {}\n'.format(name.upper(), weight_ih_l0.shape[1]//3))
    hf.write('extern const SparseGRULayer {};\n\n'.format(name));
    return True


def dump_gru_layer(self, name, f, hf):
    global max_rnn_neurons
    print("printing layer " + name + " of type " + self.__class__.__name__)

    W_0 = self.weight_ih_l0.detach().numpy()
    W0 = pk_convert_input_kernel(W_0)  # 将pytorch格式转换为keras格式

    W_1 = self.weight_hh_l0.detach().numpy()
    W1 = pk_convert_recurrent_kernel(W_1)  # 将pytorch格式转换为keras格式

    bias_ih_l0 = self.bias_ih_l0.detach().numpy().reshape(-1)
    bias_hh_l0 = self.bias_hh_l0.detach().numpy().reshape(-1)
    bias = np.concatenate((bias_ih_l0, bias_hh_l0))
    b = pk_convert_bias(bias)

    printVector(f, W0, name + '_weights')
    printVector(f, W1, name + '_recurrent_weights')
    printVector(f, b, name + '_bias')
    if hasattr(self, 'activation'):
        activation = self.activation.__name__.upper()
    else:
        activation = 'TANH'
    if hasattr(self, 'reset_after') and not self.reset_after:
        reset_after = 0
    else:
        reset_after = 1
    neurons = W0.shape[1]//3
    max_rnn_neurons = max(max_rnn_neurons, neurons)
    f.write('const GRULayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}_recurrent_weights,\n   {}, {}, ACTIVATION_{}, {}\n}};\n\n'
            .format(name, name, name, name, W0.shape[0], W0.shape[1]//3, activation, reset_after))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), W0.shape[1]//3))
    hf.write('#define {}_STATE_SIZE {}\n'.format(name.upper(), W0.shape[1]//3))
    hf.write('extern const GRULayer {};\n\n'.format(name));
    return True
nn.GRU.dump_layer = dump_gru_layer

def dump_dense_layer(self, name, f, hf):
    print("printing layer " + name + " of type " + self.__class__.__name__)
    weight = self.weight.detach().numpy()
    weight = weight.T
    bias = self.bias.detach().numpy()
    activation = 'TANH'
    dump_dense_layer_impl(name, weight, bias, activation, f, hf)
    return False
nn.Linear.dump_layer = dump_dense_layer

def dump_mdense_layer(self, name, f, hf):
    global max_mdense_tmp
    print("printing layer " + name + " of type " + self.__class__.__name__)
    weight1 = self.weight1.detach().numpy()
    weight2 = self.weight2.detach().numpy()
    weight1 = weight1.reshape(weight1.shape[0], weight1.shape[1], 1)
    weight2 = weight2.reshape(weight2.shape[0], weight2.shape[1], 1)
    weight = np.concatenate((weight1, weight2), 2)

    bias1 = self.bias1.detach().numpy()
    bias2 = self.bias2.detach().numpy()
    bias1 = bias1.reshape(bias1.shape[0], 1)
    bias2 = bias2.reshape(bias2.shape[0], 1)
    bias  = np.concatenate((bias1, bias2), 1)

    factor1 = self.factor1.detach().numpy()
    factor2 = self.factor2.detach().numpy()
    factor1 = factor1.reshape(factor1.shape[0], 1)
    factor2 = factor2.reshape(factor2.shape[0], 1)
    factor  = np.concatenate((factor1,factor2), 1)

    printVector(f, np.transpose(weight, (1, 2, 0)), name + '_weights')
    printVector(f, np.transpose(bias, (1, 0)), name + '_bias')
    printVector(f, np.transpose(factor, (1, 0)), name + '_factor')
    activation = 'SOFTMAX'
    max_mdense_tmp = max(max_mdense_tmp, weight.shape[0]*weight.shape[2])
    f.write('const MDenseLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}_factor,\n   {}, {}, {}, ACTIVATION_{}\n}};\n\n'
            .format(name, name, name, name, weight.shape[1], weight.shape[0], weight.shape[2], activation))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weight.shape[0]))
    hf.write('extern const MDenseLayer {};\n\n'.format(name));
    return False
MDense.dump_layer = dump_mdense_layer


def dump_conv1d_layer(self, name, f, hf):
    global max_conv_inputs
    print("printing layer " + name + " of type " + self.__class__.__name__)
    weight = self.weight.detach().numpy()
    weight = weight.transpose(2, 1, 0)
    bias = self.bias.detach().numpy()

    printVector(f, weight, name + '_weights')
    printVector(f, bias, name + '_bias')
    activation = 'TANH'
    max_conv_inputs = max(max_conv_inputs, weight.shape[1]*weight.shape[0])
    f.write('const Conv1DLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}, {}, {}, ACTIVATION_{}\n}};\n\n'
            .format(name, name, name, weight.shape[1], weight.shape[0], weight.shape[2], activation))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weight.shape[2]))
    hf.write('#define {}_STATE_SIZE ({}*{})\n'.format(name.upper(), weight.shape[1], (weight.shape[0]-1)))
    hf.write('#define {}_DELAY {}\n'.format(name.upper(), (weight.shape[0]-1)//2))
    hf.write('extern const Conv1DLayer {};\n\n'.format(name));
    return True
nn.Conv1d.dump_layer = dump_conv1d_layer


def dump_embedding_layer(self, name, f, hf):
    print("printing layer " + name + " of type " + self.__class__.__name__)
    weights = self.weight.detach().numpy()
    dump_embedding_layer_impl(name, weights, f, hf)
    return False
nn.Embedding.dump_layer = dump_embedding_layer



def dump_lpcnet(hparams):
    model = LPCNet(hparams)

    if None == hparams.checkpoint_file or not os.path.isfile(hparams.checkpoint_file):
        return
    checkpoint_dict = torch.load(hparams.checkpoint_file)
    model.load_state_dict(checkpoint_dict['state_dict'])

    #这里的model_init是为了测试
    #model_init(model)
    cfile = 'nnet_data.c'
    hfile = 'nnet_data.h'

    f   = open(cfile, 'w')
    hf  = open(hfile, 'w')

    f.write('/*This file is automatically generated from a Pytorch model*/\n\n')
    f.write('#ifdef HAVE_CONFIG_H\n#include "config.h"\n#endif\n\n#include "nnet.h"\n#include "{}"\n\n'.format(hfile))

    hf.write('/*This file is automatically generated from a Pytorch model*/\n\n')
    hf.write('#ifndef RNN_DATA_H\n#define RNN_DATA_H\n\n#include "nnet.h"\n\n')

    embed_size = hparams.embedding_size

    E = model.embed_sig.weight.detach().numpy()
    W = model.gru_a.weight_ih_l0.detach().numpy()
    W = pk_convert_input_kernel(W) #将pytorch格式转换为keras格式
    W1 = W[:embed_size, :]
    dump_embedding_layer_impl('gru_a_embed_sig', np.dot(E, W1), f, hf)
    W2 = W[embed_size:2*embed_size, :]
    dump_embedding_layer_impl('gru_a_embed_pred', np.dot(E, W2), f, hf)
    E1 = model.embed_exc.weight.detach().numpy()
    W3 = W[2*embed_size:3*embed_size, :]
    dump_embedding_layer_impl('gru_a_embed_exc', np.dot(E1, W3), f, hf)
    W4 = W[3*embed_size:, :]

    bias_ih_l0 = model.gru_a.bias_ih_l0.detach().numpy().reshape(-1)
    bias_hh_l0 = model.gru_a.bias_hh_l0.detach().numpy().reshape(-1)
    bias = np.concatenate((bias_ih_l0, bias_hh_l0))
    b = pk_convert_bias(bias)

    dump_dense_layer_impl('gru_a_dense_feature', W4, b, 'LINEAR', f, hf)

    #layer列表
    layer_list = []
    model_list = [model.embed_pitch, model.feature_conv1, model.feature_conv2, model.feature_dense1, model.embed_sig,
                  model.embed_exc, model.feature_dense2, model.gru_a, model.gru_b, model.md]
    model_name = ['embed_pitch', 'feature_conv1', 'feature_conv2', 'feature_dense1', 'embed_sig',
                  'embed_exc', 'feature_dense2', 'gru_a', 'gru_b', 'dual_fc']
    #for i, layer in enumerate(model.named_children()):
    for idx in range(len(model_list)):
        name = model_name[idx]
        layer = model_list[idx]
        print("----------------" + name + "--------------------")
        if layer.dump_layer(name, f, hf):
            layer_list.append(name)

    dump_sparse_gru(model.gru_a, f, hf)

    hf.write('#define MAX_RNN_NEURONS {}\n\n'.format(max_rnn_neurons))
    hf.write('#define MAX_CONV_INPUTS {}\n\n'.format(max_conv_inputs))
    hf.write('#define MAX_MDENSE_TMP {}\n\n'.format(max_mdense_tmp))

    hf.write('typedef struct {\n')
    for i, name in enumerate(layer_list):
        hf.write('  float {}_state[{}_STATE_SIZE];\n'.format(name, name.upper()))
    hf.write('} NNetState;\n')

    hf.write('\n\n#endif\n')

    f.close()
    hf.close()


if __name__ == "__main__":

    hparams = create_hparams()
    dump_lpcnet(hparams)