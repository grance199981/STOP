import logging
import shutil
import os
import random

import torch
import numpy as np
import torch.nn.functional as F

def setup_logging(log_file='log.txt'):
    """Setup logging configuration"""
    # 创建一个 logger 对象
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # 创建一个文件处理器，将日志写入指定的文件中
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    # 创建一个格式化器，定义日志的格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 将文件处理器添加到 logger 对象中
    logger.addHandler(file_handler)

    return logger

    # logging.basicConfig(level=logging.DEBUG,
    #                     format="%(asctime)s - %(levelname)s - %(message)s",
    #                     datefmt="%Y-%m-%d %H:%M:%S",
    #                     filename=log_file,
    #                     filemode='w',
    #                     )
    #
    # if not os.access(log_file, os.W_OK):
    #     print(f"没有写入权限: {log_file}")
    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(message)s')
    # console.setFormatter(formatter)
    # logging.getLogger('').addHandler(console)
    
def save_checkpoint(state, is_best, save_path):
    torch.save(state, os.path.join(save_path, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(save_path, 'checkpoint.pth.tar'), os.path.join(save_path, 'best.pth.tar'))
        

def lr_scheduler(optimizer, epoch, lr_decay_epoch=50, decay_factor=0.1):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_factor
    return optimizer

def count_parameters(model):
    ''' Count number of parameters in model influenced by global loss. '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_one_hot(y, n_dims=None):
    ''' Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. '''
    y_tensor = y.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot

def reproducible_config(seed=1234, is_cuda=False):
    """Some configurations for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if is_cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(seed)

def average_pooling_through_time(x, time_window):
    for step in range(time_window):
        if step == 0:
            output = F.avg_pool2d(x[step], 2)
            output_return = output.clone()
        else:
            output = F.avg_pool2d(x[step], 2)
            output_return = torch.cat((output_return, output), dim=0)
    return output_return.view(-1, *output.size())

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res