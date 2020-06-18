import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *
import time
import torch.optim as optim
import torch.nn.functional as F

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=19,
                    help='depth of the vgg')
parser.add_argument('--num_sample', type=int, default=100,
                    help='number of samples for FSKD')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')


def main():
    global args
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model = vgg(dataset=args.dataset, depth=args.depth, save_feature=True, batch_norm=False)
    origin_model = vgg(dataset=args.dataset, depth=args.depth, save_feature=True, batch_norm=False)

    num_parameters = sum([param.nelement() for param in origin_model.parameters()])
    print("Origin number of parameters: {}".format(num_parameters))

    if args.cuda:
        model.cuda()
        origin_model.cuda()

    if args.model:
        if os.path.isfile(args.model):
            print("=> loading checkpoint '{}'".format(args.model))
            checkpoint = torch.load(args.model)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            origin_model.load_state_dict(checkpoint['state_dict'])

            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                  .format(args.model, checkpoint['epoch'], best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    PRUNED_CHANNEL = [
        0.5, 0,
        0, 0,
        0, 0, 0,
        0.5, 0.5, 0.5,
        0.5, 0.5, 0.5
    ]

    cfg = []
    cfg_mask = []

    layer_id = 0

    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            weight_copy = m.weight.data.cpu().numpy()
            weight_norm = np.sum(np.abs(weight_copy), axis=(1, 2, 3))
            num_channel = len(weight_norm)
            if PRUNED_CHANNEL[layer_id] == 0:
                thre = -1
            else:
                thre = sorted(weight_norm)[int(num_channel * PRUNED_CHANNEL[layer_id]) - 1]
            mask = (weight_norm > thre).astype(np.int64)
            cfg.append(int(np.sum(mask)))
            cfg_mask.append(mask)
            layer_id += 1
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, len(mask), int(np.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    print('Pre-processing Successful!')

    # Make real prune
    print(cfg)
    newmodel = vgg(dataset=args.dataset, cfg=cfg, save_feature=True, batch_norm=False)
    if args.cuda:
        newmodel.cuda()
    num_parameters_new = sum([param.nelement() for param in newmodel.parameters()])
    print("New number of parameters: {}".format(num_parameters_new))
    print("Parameter pruning: {}".format(1-num_parameters_new/num_parameters))

    layer_id_in_cfg = 0
    start_mask = np.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask)))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            start_mask = end_mask
            layer_id_in_cfg += 1
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask)))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

    torch.save({'cfg_mask': cfg_mask, 'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))
    # print(newmodel)
    model = newmodel
    test(model)
    model.add_pwconv(batch_norm=False)
    print("Origin model:")
    test(origin_model)
    print("Pruned model before recover:")
    test(model)

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    if args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data.cifar10', train=True, transform=transforms.Compose([
                # transforms.Pad(4),
                # transforms.RandomCrop(32, 4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.num_sample, shuffle=False,
            num_workers=0, pin_memory=False)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data.cifar100', train=True, transform=transforms.Compose([
                # transforms.Pad(4),
                # transforms.RandomCrop(32, 4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.num_sample, shuffle=False,
            num_workers=0, pin_memory=False)

    cfg_mask_idx = 0
    cpu_time = 0
    start_time = time.time()
    for m in origin_model.modules():
        if isinstance(m, nn.Conv2d):
            mask = np.squeeze(np.argwhere(np.asarray(cfg_mask[cfg_mask_idx])))
            _, one_cpu_time = recover_one_layer(model, origin_model, num_sample=args.num_sample, layer_idx=cfg_mask_idx, mask=mask, train_loader=train_loader)
            cfg_mask_idx += 1
            cpu_time += one_cpu_time

    print("Pruned model before absorb:")
    test(model)
    model.absorb_pwconv(batch_norm=False)
    # model.add_pwconv(batch_norm=False)
    # mask = np.squeeze(np.argwhere(np.asarray(cfg_mask[15].cpu().numpy())))
    # recover_one_layer(newmodel, origin_model, num_sample=500, layer_idx=15, mask=mask)
    print("Total time: {:.3f}s".format(time.time() - start_time))
    print("CPU time: {:.3f}s".format(cpu_time))
    print("Pruned model after absorb:")
    test(model)


def recover_one_layer(new_model, origin_model, num_sample, layer_idx, mask, train_loader):
    recover_time = time.time()

    # Data loading code

    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    new_model.eval()
    origin_model.eval()

    end = time.time()

    sample_count = 0

    for i, (input, target) in enumerate(train_loader):
        # if sample_each_class[target.numpy()[0]] >= num_sample_per_class:
        #     continue
        if sample_count >= num_sample:
            break

        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(input).cuda()

        # compute output
        origin_model(input_var)
        new_model(input_var)

        # extract feature
        C_origin = origin_model.inter_feature[layer_idx].size(1)
        C_new = new_model.inter_feature[layer_idx].size(1)
        origin_feature = origin_model.inter_feature[layer_idx].permute(0, 2, 3, 1).contiguous().view(-1, C_origin).data.cpu().numpy().astype(np.float32)
        new_feature = new_model.inter_feature[layer_idx].permute(0, 2, 3, 1).contiguous().view(-1, C_new).data.cpu().numpy().astype(np.float32)
        # if i == 0:
        #     WH = int(np.sqrt(len(origin_feature) / args.num_sample))
        #     origin_output = np.zeros((WH * WH * num_sample, C_origin))
        #     new_output = np.zeros((WH * WH * num_sample, C_new))
        origin_output = origin_feature
        new_output = new_feature
        sample_count += args.num_sample

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    cpu_time = time.time()
    origin_output = origin_output[:, mask.tolist()]

    ret = np.linalg.lstsq(new_output, origin_output, rcond=None)
    x = ret[0]
    # print("Distance before: {}. Distance after: {}".format(
    #     np.linalg.norm(origin_output - new_output) / (np.shape(origin_output)[0] * np.shape(origin_output)[1]),
    #     np.linalg.norm(np.dot(new_output, x) - origin_output) / (np.shape(origin_output)[0] * np.shape(origin_output)[1])))
    x = np.transpose(x)
    new_model.pwconv[layer_idx].weight.data.copy_(torch.from_numpy(x).view(C_new, C_new, 1, 1))
    print("Reocver from layer {} takes {}s".format(layer_idx, time.time() - recover_time))
    return new_model, time.time() - cpu_time


# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.3f})\n'.format(
        correct, len(test_loader.dataset), 100. * correct.float() / len(test_loader.dataset)))
    return correct


class AverageMeter(object):
    """Computes and stores the average and current value"""
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
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    with torch.no_grad():
        main()
