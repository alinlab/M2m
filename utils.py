'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
from datetime import datetime
import shutil
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import importlib

def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sum_t(tensor):
    return tensor.float().sum().item()


class InputNormalize(nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed
    mean and standard deviation (user-specified).
    '''
    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None].cuda()
        new_mean = new_mean[..., None, None].cuda()

        # To prevent the updates the mean, std
        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean)/self.new_std
        return x_normalized


class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""
    def __init__(self, fn):
        if not os.path.exists("./logs/"):
            os.mkdir("./logs/")

        logdir = 'logs/' + fn
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        if len(os.listdir(logdir)) != 0:
            ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
                            "Will you proceed [y/N]? ")
            if ans in ['y', 'Y']:
                shutil.rmtree(logdir)
            else:
                exit(1)
        self.set_dir(logdir)

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def log(self, string):
        self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
        self.log_file.flush()

        print('[%s] %s' % (datetime.now(), string))
        sys.stdout.flush()

    def log_dirname(self, string):
        self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
        self.log_file.flush()

        print('%s (%s)' % (string, self.logdir))
        sys.stdout.flush()


######## Loss ########


def soft_cross_entropy(input, labels, reduction='mean'):
    xent = (-labels * F.log_softmax(input, dim=1)).sum(1)
    if reduction == 'sum':
        return xent.sum()
    elif reduction == 'mean':
        return xent.mean()
    elif reduction == 'none':
        return xent
    else:
        raise NotImplementedError()


def classwise_loss(outputs, targets):
    out_1hot = torch.ones_like(outputs)
    out_1hot.scatter_(1, targets.view(-1, 1), -1)
    return (outputs * out_1hot).mean()


def focal_loss(input_values, gamma):
    """Computes the focal loss
    
    Reference: https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
    """
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss


class FocalLoss(nn.Module):
    """Reference: https://github.com/kaidic/LDAM-DRW/blob/master/losses.py"""
    def __init__(self, weight=None, gamma=0., reduction='mean'):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, weight=self.weight, reduction=self.reduction), self.gamma)


class LDAMLoss(nn.Module):
    """Reference: https://github.com/kaidic/LDAM-DRW/blob/master/losses.py"""
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, reduction='mean'):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.scale = s
        self.weight = weight
        self.reduction = reduction

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.scale * output, target, weight=self.weight, reduction=self.reduction)

######## Generation ########


def project(inputs, orig_inputs, attack, eps):
    diff = inputs - orig_inputs
    if attack == 'l2':
        diff = diff.renorm(p=2, dim=0, maxnorm=eps)
    elif attack == 'inf':
        diff = torch.clamp(diff, -eps, eps)
    return orig_inputs + diff


def make_step(grad, attack, step_size):
    if attack == 'l2':
        grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
        scaled_grad = grad / (grad_norm + 1e-10)
        step = step_size * scaled_grad
    elif attack == 'inf':
        step = step_size * torch.sign(grad)
    else:
        step = step_size * grad
    return step


def random_perturb(inputs, attack, eps):
    if attack == 'inf':
        r_inputs = 2 * (torch.rand_like(inputs) - 0.5) * eps
    else:
        r_inputs = (torch.rand_like(inputs) - 0.5).renorm(p=2, dim=1, maxnorm=eps)
    return r_inputs


######## Data ########


def make_imb_data(max_num, min_num, class_num, gamma):
    class_idx = torch.arange(1, class_num + 1).float()
    ratio = max_num / min_num
    b = (torch.pow(class_idx[-1], gamma) - ratio) / (ratio - 1)
    a = max_num * (1 + b)
    class_num_list = []
    for i in range(class_num):
        class_num_list.append(int(torch.round(a / (torch.pow(class_idx[i], gamma) + b))))
    print(class_num_list)

    return list(class_num_list)


def make_imb_data2(max_num, class_num, gamma):
    mu = np.power(1/gamma, 1/(class_num - 1))
    print(mu)
    class_num_list = []
    for i in range(class_num):
        class_num_list.append(int(max_num * np.power(mu, i)))

    return list(class_num_list)


def inf_data_gen(dataloader):
    while True:
        for images, targets in dataloader:
            yield images, targets


def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 50.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
