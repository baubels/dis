
import torch
import numpy as np


def log10(x):
    """ Safe base-10 log of x. """
    return torch.log10(torch.clamp(x, min=1e-10))


class Result(object):
    """ Stores the result of a (output, target)[(batch_size, 5)] Qauternion + distance loss calculations.

    Evaluates:
    - ang1 (angular distance between two unit quaternions)

    explanation of each loss: y is true, y* is pred
    ang1: 1/N arccos(2*<p,q>^2 - 1)
    """

    def __init__(self):
        self.ang1 = 0
        self.dis1 = 0
        self.data_time, self.gpu_time = 0, 0

    def set_to_worst(self):
        self.ang1 = np.inf
        self.dis1 = np.inf
        self.data_time, self.gpu_time = 0, 0

    def update(self, ang, dis, gpu_time, data_time):
        self.ang1 = ang
        self.dis1 = dis
        self.data_time, self.gpu_time = data_time, gpu_time

    def evaluate(self, output, target):
        # redefines to just include angular distance
        angular_output = torch.nn.functional.normalize(output[:, :4], p=2, dim=1)
        angular_target = torch.nn.functional.normalize(target[:, :4], p=2, dim=1)
        
        inner_prod_square = torch.square(torch.sum(angular_output*angular_target, dim=1))
        theta = torch.arccos(2*inner_prod_square-1)
        self.ang1 = theta.mean()

        distance_output = output[:, 4:]
        distance_target = target[:, 4:]

        distance_mse = torch.nn.functional.mse_loss(distance_output, distance_target)
        self.dis1 = distance_mse.mean()
        
        self.data_time = 0
        self.gpu_time = 0


class AverageMeter(object):
    """ Usage: aggregate `Result` estimate metrics across an epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 1.0
        self.sum_ang1 = 0
        self.sum_dis1 = 0
        self.sum_data_time, self.sum_gpu_time = 0, 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n
        self.sum_ang1 += n * result.ang1
        self.sum_dis1 += n * result.dis1
        self.sum_data_time += n * data_time
        self.sum_gpu_time += n * gpu_time

    def average(self):
        avg = Result()
        avg.update(
            self.sum_ang1 / self.count, 
            self.sum_dis1 / self.count,
            self.sum_gpu_time / self.count, 
            self.sum_data_time / self.count)
        return avg
