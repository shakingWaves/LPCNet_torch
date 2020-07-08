# LPCNet



Low complexity implementation of the WaveRNN-based LPCNet algorithm, as described in:

J.-M. Valin, J. Skoglund, [LPCNet: Improving Neural Speech Synthesis Through Linear Prediction](https://jmvalin.ca/papers/lpcnet_icassp2019.pdf), *Submitted for ICASSP 2019*, arXiv:1810.11846.

# Warning
Tacotron2 or DeepVoice3 which predictes parameters including ceptral coefficients and 2 pitch parameters undiscriminating is probably not good enough to estimate pitch parameters. However, LPCNet is sensitive to the estimation of pitch coefficents, so it is not recommended to predict the features when training acotron2 or DeepVoice3 replacing mel spectrum with ceptral coefficients and 2 pitch parameters directly.    
Update:
Training your tacotron2 or DeepVoice3 very well may help to estimate the pitch parameters.

# Introduction

Work in progress software for researching low CPU complexity algorithms for speech synthesis and compression by applying Linear Prediction techniques to WaveRNN. High quality speech can be synthesised on regular CPUs (around 3 GFLOP) with SIMD support (AVX, AVX2/FMA, NEON currently supported).

The BSD licensed software is written in C and Python/Keras. For training, a GTX 1080 Ti or better is recommended.

This software is an open source starting point for WaveRNN-based speech synthesis and coding.

__NOTE__: The repo aims to work with Tacotron2.

# Quickstart

1. Set up a Keras system with GPU.

1. Generate training data:
   ```
   make dump_data
   ./dump_data -train input.s16 features.f32 data.u8
   ```
   where the first file contains 16 kHz 16-bit raw PCM audio (no header) and the other files are output files. This program makes several passes over the data with different filters to generate a large amount of training data.

1. Now that you have your files, train with:
   ```
   ./train_lpcnet.py features.f32 data.u8
   ```
   and it will generate a wavenet*.h5 file for each iteration. If it stops with a 
   "Failed to allocate RNN reserve space" message try reducing the *batch\_size* variable in train_wavenet_audio.py.

1. You can synthesise speech with Python and your GPU card:
   ```
   ./dump_data -test test_input.s16 test_features.f32
   ./test_lpcnet.py test_features.f32 test.s16
   ```
   Note the .h5 is hard coded in test_lpcnet.py, modify for your .h file.

1. Or with C on a CPU:
   First extract the model files nnet_data.h and nnet_data.c
   ```
   ./dump_lpcnet.py lpcnet15_384_10_G16_64.h5
   ```
   Then you can make the C synthesiser and try synthesising from a test feature file:
   ```
   make test_lpcnet
   ./dump_data -test test_input.s16 test_features.f32
   ./test_lpcnet test_features.f32 test.s16
   ffmpeg -f s16le -ar 16k -ac 1 -i test.s16 test-out.wav
   ```
 
# Speech Material for Training LPCNet

Suitable training material can be obtained from the [McGill University Telecommunications & Signal Processing Laboratory](http://www-mmsp.ece.mcgill.ca/Documents/Data/).  Download the ISO and extract the 16k-LP7 directory, the src/concat.sh script can be used to generate a headerless file of training samples.
```
cd 16k-LP7
sh /path/to/concat.sh
```

# Speech Material for Training Tacotron2
Although the model has 55 dims features when training LPCNet, there are 20 features to be used as input features when inferring the audio. You should enble TACOTRON2 Macro in Makefile to get the features for Training Tacotron2. You also should generate indepent features for every audio when training Tacotron2 other than concatate all features into one file when training LPCNet.
```bash
#preprocessing
./header_removal.sh
make dump_data taco=1   # Define TACOTRON2 macro
./feature_extract.sh
```
```bash
#synthesis
make test_lpcnet taco=1 # Define TACOTRON2 macro
./test_lpcnet test_features.f32 test.s16
ffmpeg -f s16le -ar 16k -ac 1 -i test.s16 test-out.wav
```

# How to combine the LPCNet and Tacotron2.  
## When training  
the Materials are generated independently with two procedures.
1. for LPCNet   
* Remove the header of audio files and prepare the concatated materials.
```bash
cd 16k-LP7
sh /path/to/concat.sh
```
* Build the repo without TACOTRON2 Macro and generate the data for LPCNet. It will follow the Mozilla version totally.   
```bash
make dump_data
./dump_data -train input.s16 features.f32 data.u8
```
* Train the LPCNet.   
```bash
./train_lpcnet.py features.f32 data.u8
``` 

2. For tacotron2    
You can git clone the [repo](https://github.com/Rayhane-mamah/Tacotron-2) to as the front-end.    
* Re-Build the repo of LPCNet from the floor. If you want to use the LPCNet in (1) step, plz make clean first as follows. otherwise you can skip the step.
```bash
make clean
```
* Build the repo of LPCnet with TACOTRON2 Macro    
```bash
make dump_data taco=1   # Define TACOTRON2 macro
```
* Remove the header of audio files and generate the features for Tacotron2 but NOT concatate the data with my scripts. you should replace the direcroty in the scripts with your data folder.
```bash
./header_removal.sh
./feature_extract.sh
```
* Convert the data generated at the last step which has .f32 extension to what could be loaded with numpy. I merge it to the Tacotron feeder [here](https://github.com/Rayhane-mamah/Tacotron-2/blob/master/tacotron/feeder.py#L192) and [here](https://github.com/Rayhane-mamah/Tacotron-2/blob/master/tacotron/feeder.py#L128) with the following code.
```python
mel_target = np.fromfile(os.path.join(self._mel_dir, meta[0]), dtype='float32')
mel_target = np.resize(mel_target, (-1, self._hparams.num_mels))
```
* train the tacotron2 with the features  and follow the steps of [Tacatron2 repo](https://github.com/Rayhane-mamah/Tacotron-2). You need to modify the dimensions of input features(num_mels) to 20.

## When synthesising.
1. synthesis the features with Tacotron2 following the steps of [Tacatron2 repo](https://github.com/Rayhane-mamah/Tacotron-2)   
2. convert the features with npy format to \*.f32 format with the following python scripts.
```python
import numpy as np
npy_data = np.load("npy_from_tacotron2.npy")
npy_data = npy_data.reshape((-1,))
npy_data.tofile("f32_for_lptnet.f32")
```
3. Synthesis the waveform using LPCNet
You should  use LPCNet of the C code version and make sure that the LPCNet is builded with TACOTRON2 Macro.    
* First extract the model files nnet_data.h and nnet_data.c
   ```bash
   ./dump_lpcnet.py lpcnet15_384_10_G16_64.h5
   ```
* Build the test LPCNet with TACOTRON2 Macro.   
``` bash
make test_lpcnet taco=1 # Define TACOTRON2 macro
```
* synthesis theraw data of waveform using the LPCNet of the C code version.
``` bash
./test_lpcnet f32_for_lptnet.f32 test.s16
```
* add the headder of audio the raw data and generate the pcm audio.
``` bash
ffmpeg -f s16le -ar 16k -ac 1 -i test.s16 test-out.wav
```

# Reading Further

1. [LPCNet: DSP-Boosted Neural Speech Synthesis](https://people.xiph.org/~jm/demo/lpcnet/)
2. Sample model files:
https://jmvalin.ca/misc_stuff/lpcnet_models/

