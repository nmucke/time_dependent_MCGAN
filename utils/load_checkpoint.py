import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch

def load_checkpoint(checkpoint_path, encoder=None, decoder=None, critic=None,
                    encoder_optimizer=None, encoder_reg_optimizer=None,
                    decoder_optimizer=None,
                      critic_optimizer=None):
    checkpoint = torch.load(checkpoint_path)

    if encoder_optimizer is not None:
        encoder_optimizer.load_state_dict(
                checkpoint['encoder_optimizer_state_dict'])
    if encoder_reg_optimizer is not None:
        encoder_optimizer.load_state_dict(
                checkpoint['encoder_reg_optimizer_state_dict'])
    if decoder_optimizer is not None:
        decoder_optimizer.load_state_dict(
                checkpoint['decoder_optimizer_state_dict'])
    if critic_optimizer is not None:
        critic_optimizer.load_state_dict(
                checkpoint['critic_optimizer_state_dict'])

    if encoder is not None:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
    if decoder is not None:
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
    if critic is not None:
        critic.load_state_dict(checkpoint['critic_state_dict'])