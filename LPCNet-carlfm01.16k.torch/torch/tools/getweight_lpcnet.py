#!/usr/bin/python3
'''Copyright (c) 2018 Mozilla

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import lpcnet
import sys
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from ulaw import ulaw2lin, lin2ulaw
import keras.backend as K
import h5py

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

model, enc, dec = lpcnet.new_lpcnet_model()
model.load_weights('lpcnet20_384_10_G16_27.h5')

embed_exc = model.get_layer('embed_exc')
exc_weight = embed_exc.get_weights()
w = exc_weight[0]
np.savetxt("../data/output/embed_exc_weight.txt", w)

embed_pitch = model.get_layer('embed_pitch')
pitch_weight = embed_pitch.get_weights()
w = pitch_weight[0]
np.savetxt("../data/output/embed_pitch_weight.txt", w)



embed_sig = model.get_layer('embed_sig')
sig_weight = embed_sig.get_weights();
w = sig_weight[0]
np.savetxt("../data/output/embed_sig_weight.txt", w)


feature_conv1 = model.get_layer('feature_conv1')
conv1_weight = feature_conv1.get_weights()
w0= conv1_weight[0]
w0 = w0.reshape(3, -1)
np.savetxt("../data/output/feature_conv1_weight.txt", w0)
w1 = conv1_weight[1]
np.savetxt("../data/output/feature_conv1_bias.txt", w1)


feature_conv2 = model.get_layer('feature_conv2')
conv2_weight = feature_conv2.get_weights()
w0= conv2_weight[0]
w0 = w0.reshape(3, -1)
np.savetxt("../data/output/feature_conv2_weight.txt", w0)
w1 = conv2_weight[1]
np.savetxt("../data/output/feature_conv2_bias.txt", w1)



feature_dense1 = model.get_layer('feature_dense1')
dense1_weight = feature_dense1.get_weights()
w0 = dense1_weight[0]
np.savetxt("../data/output/feature_dense1_weight.txt", w0)
w1 = dense1_weight[1]
np.savetxt("../data/output/feature_dense1_bias.txt", w1)

feature_dense2 = model.get_layer('feature_dense2')
dense2_weight = feature_dense2.get_weights()
w0 = dense2_weight[0]
np.savetxt("../data/output/feature_dense2_weight.txt", w0)
w1 = dense2_weight[1]
np.savetxt("../data/output/feature_dense2_bias.txt", w1)

gru_a = model.get_layer('gru_a')
a_weight = gru_a.get_weights()
w0 = a_weight[0]
np.savetxt("../data/output/gru_a_ih_weight.txt", w0)
w1 = a_weight[1]
np.savetxt("../data/output/gru_a_hh_weight.txt", w1)
w2 = a_weight[2]
np.savetxt("../data/output/gru_a_bias.txt", w2)


gru_b = model.get_layer('gru_b')
b_weight = gru_b.get_weights()
w0 = b_weight[0]
np.savetxt("../data/output/gru_b_ih_weight.txt", w0)
w1 = b_weight[1]
np.savetxt("../data/output/gru_b_hh_weight.txt", w1)
w2 = b_weight[2]
np.savetxt("../data/output/gru_b_bias.txt", w2)



dual_fc = model.get_layer('dual_fc')
fc_weight = dual_fc.get_weights()
w0 = fc_weight[0][:,:,0]
w1 = fc_weight[0][:,:,1]
b0 = fc_weight[1][:,0]
b1 = fc_weight[1][:,1]
f0 = fc_weight[2][:,0]
f1 = fc_weight[2][:,1]
np.savetxt("../data/output/dual_fc_w0_weight.txt", w0)
np.savetxt("../data/output/dual_fc_w1_weight.txt", w1)
np.savetxt("../data/output/dual_fc_b0_weight.txt", b0)
np.savetxt("../data/output/dual_fc_b1_weight.txt", b1)
np.savetxt("../data/output/dual_fc_f0_weight.txt", f0)
np.savetxt("../data/output/dual_fc_f1_weight.txt", f1)


