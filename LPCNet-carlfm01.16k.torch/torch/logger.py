import random
import torch
from torch.utils.tensorboard import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy


class LPCNetLogger(SummaryWriter):
    def __init__(self, logdir):
        super(LPCNetLogger, self).__init__(logdir)

    def log_training(self, label ,reduced_loss, avg_loss, learning_rate, duration,
                     iteration):
        self.add_scalar(label + '/training.loss', reduced_loss, iteration)
        self.add_scalar(label + '/avg.loss', avg_loss, iteration)
        self.add_scalar(label + "/learning.rate", learning_rate, iteration)
        self.add_scalar(label + "/duration", duration, iteration)

    def log_validation(self, reduced_loss, iteration):
        self.add_scalar("Validate/validation.loss", reduced_loss, iteration)
