import torch
import torch.nn as nn
from torch.nn import functional as F


class CourageLoss(nn.Module):
    def __init__(self, gamma, weight=None, ignore_index=-100, cl_eps=1e-5):
        super(CourageLoss, self).__init__()
        self.gamma = gamma
        self.num_class = 25  # TODO
        self.ignore_index = ignore_index
        self.weight = cal_effective_weight([2889, 1139, 2961, 2376, 1835, 1123, 3753, 3210, 3140, 1521, 11243, 6651, 10870, 11123, 7455, 6711, 9194, 9059, 5715, 5225, 294549, 2, 17, 2, 1])
        # self.weight = [weight]  # shape [num_classes]
        self.cl_eps = cl_eps

    def forward(self, x, target, is_train=True):
        lprobs = F.log_softmax(x, dim=1)
        mle_loss = F.nll_loss(lprobs, target, reduction='mean', ignore_index=self.ignore_index)  # -y* log p
        org_loss = mle_loss
        # if is_train and not (self.opt.defer_start and self.get_epoch() <= self.opt.defer_start):  # defer encourage loss
        if is_train:
            probs = torch.exp(lprobs)
            bg = self.gamma
            if bg > 0:
                bonus = -torch.pow(probs, bg)  # power bonus
            else:
                bonus = torch.log(torch.clamp((torch.ones_like(probs) - probs), min=self.cl_eps)) # likelihood bonus
            # weight_courage = self.weight / torch.max(self.weight) # bounded in [0,1]
            weight_courage = self.weight  # unbounded
            c_loss = F.nll_loss(
                -bonus * weight_courage,
                target.view(-1),
                reduction='mean',
                ignore_index=self.ignore_index,
            )  # y*log(1-p)
            all_loss = mle_loss + c_loss
        else:
            all_loss = mle_loss
        return all_loss

    # def set_epoch(self, epoch_i):
    #     self.courage_items['epoch_i'] = epoch_i + 1  # 从1 开始计数
    # def get_epoch(self):
    #     return  self.courage_items['epoch_i']


def cal_effective_weight(cls_num_list, beta=0.9999):
    # cls_num_list frequency of each class, shape:[num_classes]
    # to calculate weight
    import numpy as np
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    weight = torch.FloatTensor(per_cls_weights).cuda()
    return weight
