import tensorflow as tf
def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=120,
        frame_size=480,
        bfcc_band = 25,
        nb_features= 69, #nb_features = bfcc_band * 2 + 3 + 16,
        nb_used_features=69, #bfcc_band * 2 + pitch + pitch corr
        pitch_idx = 50, # pitch_idx = 2 * bfcc_band，[0, 2*bfcc_band）是特征，2bfcc_band,2bfcc_band+1,2bfcc_band+2是基音周期相关
        pitch_max_period = 768, #48k是768, 16k是256
        shuffle = False,    #这个参数必须设置为False
        embedding_size = 128,
        embedding_pitch_size = 64,
        rnn_units1 = 384,
        rnn_units2 = 16,

    ################################
    # Data Parameters             #
    ################################
    training_files = '/data/lixiaobo5/Synthesis/LPCNet-carlfm01.48k.torch/torch/training_data/train.txt',
    validation_files = '/data/lixiaobo5/Synthesis/LPCNet-carlfm01.48k.torch/torch/training_data/validate.txt',
    test_files = '/data/lixiaobo5/Synthesis/LPCNet-carlfm01.48k.torch/torch/training_data/test.txt',
    checkpoint_path = '/data/lixiaobo5/Synthesis/LPCNet-carlfm01.48k.torch/torch/out_dir',
    checkpoint_file= '/data/lixiaobo5/Synthesis/LPCNet-carlfm01.48k.torch/torch/out_dir/pytorch_lpcnet20_384_10_G16_24.h5',
    log_dir    = '/data/lixiaobo5/Synthesis/LPCNet-carlfm01.48k.torch/torch/log',

        ################################
        # Optimization Hyperparameters #
        ################################
        learning_rate=1e-3,
        lr_decay=5e-6,
        grad_clip_thresh=1.0,
        batch_size=1,
        batch_chunk=32,
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
