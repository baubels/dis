
import torch
import torch.nn as nn


class MSELoss(nn.Module): # <- this is probably _not_ indicative of a rotation angle distance
                          # because (unit) quaternions -q and q are the same rotation but are a distance 2 away.
    """Take the MSE between two Quaternions."""
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diff = target - pred
        self.loss = (diff ** 2).mean()
        return self.loss


class MSELossDistance(nn.Module):
    """Take the MSE between two distance (shape: (batch_size, 1) predictions.)"""
    def __init__(self):
        super(MSELossDistance, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diff = target - pred
        self.loss = (diff ** 2).mean()
        return self.loss


# class RotationDistance(nn.Module):
#     """Compute the rotation angle (distance) between two Quaternions."""
#     def __init__(self):
#         super(RotationDistance, self).__init__()

#     def forward(self, pred, target):
#         assert pred.dim() == target.dim(), "inconsistent dimensions"
#         ndims = len(pred.shape) # see if input is batched or not

#         # first, normalise each quaternion
#         pred   = torch.nn.functional.normalize(pred, p=2, dim=ndims-1)
#         target = torch.nn.functional.normalize(target, p=2, dim=ndims-1) # <- targets are already normalised

#         # compute angular distance
#         inner_prod_square = torch.square(torch.sum(pred*target, dim=ndims-1))
#         theta = torch.arccos(2*inner_prod_square-1)
#         self.loss = theta.mean()
#         return self.loss


# class RotAndDist(nn.Module):
#     def __init__(self, lam: float = 0.5):
#         super(RotAndDist, self).__init__()
#         self.lambda_ = lam

#     def forward(self, pred, target):
#         assert pred.dim() == target.dim(), "inconsistent dimensions"
#         if pred.dim() >= 2:
#             rot_pred, dist_pred = pred[:, :4], pred[:, 4:]
#             rot_target, dist_target = target[:, :4], target[:, 4:]
#         else:
#             rot_pred, dist_pred = pred[:4], pred[4:]
#             rot_target, dist_target = target[:4], target[4:]

#         dist = MSELossDistance()(dist_pred, dist_target)
#         rot = RotationDistance()(rot_pred, rot_target)

#         upscale = torch.max(dist)/torch.max(rot)
#         rot = rot * upscale

#         return dist + rot
