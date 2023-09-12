import os
import torch


def save_checkpoint(state, epoch, output_directory):
    """save model `state`, save as best if is best, otherwise remember the epoch.
    """
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    torch.save(state, checkpoint_filename)
