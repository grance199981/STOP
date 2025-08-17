import argparse
from datetime import datetime
import os
import logging
import math
import time
import torch
from spikingjelly.activation_based import surrogate, functional
# from spikingjelly.clock_driven import surrogate, functional
from spiking_jelly_neuron import Nature_exponential, SpikeTrace_LIF_Neuron
# import pandas as pd

from load_dataset import load_dataset
from utils import setup_logging, save_checkpoint, to_one_hot, reproducible_config, AverageMeter, accuracy

parser = argparse.ArgumentParser(description="Code for 'STOP/OSSTOP'")
parser.add_argument('--OS', action='store_true', default=False, help='OS, enable one shot')  # False True
parser.add_argument('--cuda', type=int, default=2, help='cuda (default: 0)')
parser.add_argument('--dataset', default='CIFAR10',
                    help='datasets including '
                         'CIFAR10,'
                         'CIFAR100,'
                         'DVSCIFAR10, '
                         'DVSGesture,'
                         'ImageNet')
parser.add_argument('--model', default='CIFAR10_VGG11_BN',
                    help='models including '
                         'CIFAR10(CIFAR10_VGG11_BN, CIFAR10_ResNet_BN), '
                         'CIFAR100(CIFAR100_VGG11_BN, CIFAR100_ResNet_BN),'
                         'DVSCIFAR10(DVSCIFAR10_VGG11_BN),'
                         'DVSGesture(DVSGesture_VGG11_BN),'
                         'ImageNet(ImageNet_VGG11_BN,ImageNet_MobileNetv1_BN,ImageNet_NF_ResNet)')
parser.add_argument('--bn', type=bool, default=True,
                    help='whether enable batch normalization  (default: False)')  # False True
parser.add_argument('--bias', type=bool, default=False, help='whether enable bias in each layer  (default: False)')
parser.add_argument('--surrogate-function', type=str, default='triangle', help='surrogate including: '
                                                                               'sigmoid, '
                                                                               'expontential, '
                                                                               'triangle')
parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
parser.add_argument('--time-window', type=int, default=6, help='total time steps (default: 10)')
parser.add_argument('--lr', type=float, default=1e-1, help='initial learning rate (default: 1e-1)')
parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer including: '
                                                                 'SGD, '
                                                                 'Adam')
parser.add_argument('--momentum', type=float, default=0.9, help=' momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-5, help=' weight-decay (default:4e-5)')
parser.add_argument('--scheduler', type=str, default='Cosine', help='scheduler including: '
                                                                    'Cosine, '
                                                                    'Linear')
parser.add_argument('--lr-decay-milestones', nargs='+', type=int, default=[100, 150],
                    help='decay learning rate at these milestone epochs')
parser.add_argument('--lr-decay-fact', type=float, default=0.1,
                    help='learning rate decay factor to use at milestone epochs (default: 0.1)')
parser.add_argument('--thresh', type=float, default=1.0, help='neuronal threshold (default: 1)')
parser.add_argument('--tau', type=float, default=2,
                    help='tau factor (default: 1.1 ,1 / (1 - math.exp(-2.0))')
parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1234)')
parser.add_argument('--save-path', default='./store/', type=str, help='the directory used to save the trained models')
parser.add_argument('--resume', action='store_true', default=False, help='load checkpoint.')
parser.add_argument('--print-stats', action='store_true', default=True,
                    help='print layerwise statistics during training with local loss')
parser.add_argument('--encoding', default='real', help='spike encoding: real, latency (default: real)')
parser.add_argument('--neuron-dropout', type=float, default=0.0, help='neuron_dropout (default: 0.0)')
parser.add_argument('-loss_lambda', type=float, default=0.05,
                    help='the scaling factor for the MSE term in the loss,default =0.05,1.0')
parser.add_argument('-mse_n_reg', type=bool, default=True, help='loss function setting')
parser.add_argument('-loss_means', type=float, default=1.0, help='used in the loss function when mse_n_reg=False')


def train(args, model, epoch, device, optimizer, time_window):
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    train_loss = 0
    train_acc = 0
    train_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target_onehot = to_one_hot(target, num_classes)
        target_onehot = target_onehot.to(device)

        data, target = data.to(device), target.to(device)
        if len(data.size()) == 5:
            data = data.permute(0, 2, 3, 4, 1)

        # Clear accumulated gradient
        batch_loss = 0
        optimizer.zero_grad()

        for step in range(time_window):
            if len(data.size()) == 5:
                input = data[:, :, :, :, step]
            else:
                input = data

            if step == 0:
                out_fr = model(input)
            else:
                out_fr = model(input)

            if args.OS is False:  # disable one shot learning
                # loss_sup = torch.nn.functional.mse_loss(out_fr, target_onehot.detach())
                if step == 0:
                    total_fr = out_fr.clone().detach()
                else:
                    total_fr += out_fr.clone().detach()

                loss_sup = torch.nn.functional.cross_entropy(out_fr, target)

                loss_sup.backward()

                batch_loss += loss_sup.item()
                train_loss += loss_sup * target.size(0)



            elif args.OS is True:  # disable one shot learning
                if step == 0:
                    total_fr = out_fr.clone().detach()
                else:
                    total_fr = (1 - 1. / args.tau) * total_fr.detach() + out_fr

                if step == (time_window - 1):
                    loss_sup = torch.nn.functional.cross_entropy(total_fr, target)
                    loss_sup.backward()

                    batch_loss += loss_sup.item()
                    train_loss += loss_sup * target.size(0)

        optimizer.step()

        functional.reset_net(model)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(total_fr.data, target.data, topk=(1, 5))
        losses.update(batch_loss, data.size(0))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))

        train_samples += target.numel()
        train_acc += (total_fr.argmax(1) == target).float().sum().item()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # if batch_idx %10 ==0:
        #     print("top1_acc_avg:{:.2f},top5_acc_avg:{:.2f},losses_avg:{:.4f},time_avg:{:.2f}".format(top1.avg, top5.avg, losses.avg, batch_time.avg))

    # Format and print debug string
    # loss_average = train_loss / len(train_loader.dataset)
    loss_average = train_loss / train_samples
    train_acc /= train_samples
    train_acc = train_acc * 100

    string_print = 'Train epoch={}, lr={:.2e}, loss_local={:.4f}, accuracy={:.3f}%, time_avg={:.3f}, ' \
                   'training_samples={:d},' \
                   'mem={:.0f}MiB, max_mem={:.0f}MiB\n'.format(
        epoch,
        optimizer.state_dict()['param_groups'][0]['lr'],
        loss_average,
        train_acc,
        batch_time.avg,
        train_samples,
        torch.cuda.memory_allocated() / 1e6,
        torch.cuda.max_memory_allocated() / 1e6)

    # To store layer-wise errors
    train_acc_list = []
    if args.print_stats:
        print(string_print)
        train_acc_list.append(train_acc)

    logging.info(string_print)
    train_acc_list.append(train_acc)
    return loss_average, train_acc, train_acc_list


def test(args, model, device, time_window):
    # Change to the evaluation mode
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    test_loss = 0
    test_acc = 0
    test_samples = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            target_onehot = to_one_hot(target, num_classes)
            # data, target = data.cuda(), target.cuda()
            # target_onehot = target_onehot.cuda()
            data, target = data.to(device), target.to(device)
            target_onehot = target_onehot.to(device)
            if len(data.size()) == 5:
                data = data.permute(0, 2, 3, 4, 1)
            total_loss = 0

            # clear up the membrane
            functional.reset_net(model)

            for step in range(time_window):
                if len(data.size()) == 5:
                    input = data[:, :, :, :, step]
                else:
                    input = data

                if step == 0:
                    out_fr = model(input)
                else:
                    out_fr = model(input)

                if args.OS is False:  # disable one shot learning
                    if step == 0:
                        total_fr = out_fr.clone().detach()
                    else:
                        total_fr += out_fr.clone().detach()
                    if args.print_stats:
                        loss_sup = torch.nn.functional.mse_loss(out_fr, target_onehot.detach())
                        total_loss += loss_sup

                elif args.OS is True:  # disable one shot learning
                    if step == 0:
                        total_fr = out_fr.clone().detach()
                    else:
                        total_fr = (1 - 1. / args.tau) * total_fr.detach() + out_fr

                    if step == (time_window - 1):
                        loss_sup = torch.nn.functional.cross_entropy(total_fr, target)
                        total_loss += loss_sup

            # clear up the membrane
            functional.reset_net(model)
            test_samples += target.numel()
            test_loss += total_loss.item() * target.numel()
            test_acc += (total_fr.argmax(1) == target).float().sum().item()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(total_fr.data, target.data, topk=(1, 5))
            losses.update(total_loss, data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # if batch_idx % 100 == 0:
            #     print("top1_acc_avg:{:.2f},top5_acc_avg:{:.2f},losses_avg:{:.4f},time_avg:{:.2f}".format(top1.avg, top5.avg,
            #                                                                                      losses.avg,
            #                                                                                      batch_time.avg))

    test_loss /= test_samples
    test_acc /= test_samples
    test_acc = test_acc * 100

    # Format and print debug string
    string_print = 'Test accuracy={:.3f}%\n'.format(test_acc)

    # To store layer-wise errors
    err_list = []
    if args.print_stats:
        print(string_print)
        # states, err = model.print_stats()
        # string_print += states
        err_list.append(test_acc)
    logging.info(string_print)
    err_list.append(test_acc)
    return test_acc, err_list


if __name__ == '__main__':
    args = parser.parse_args()
    # Create dir for saving models
    save_path = args.save_path + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Logging settings
    logging = setup_logging(os.path.join(save_path, 'log.txt'))
    logging.info('args:' + str(args))
    logging.info('saving to:' + str(save_path))

    learing_way = "STOP"
    if args.OS is True:
        learing_way = "OS" + learing_way


    print("Start", learing_way)

    is_cuda = torch.cuda.is_available()
    torch.cuda.set_device(args.cuda)
    device = torch.device('cuda' if is_cuda else 'cpu')

    # Reproducibility
    reproducible_config(seed=args.seed, is_cuda=is_cuda)

    # Load datasets
    if ((args.dataset == "DVSCIFAR10") or (args.dataset == "DVSGesture")):
        train_loader, test_loader, num_classes = load_dataset(args=args, dataset=args.dataset,
                                                              batch_size=args.batch_size,
                                                              dataset_path='/home0/EDA/dataset', is_cuda=is_cuda)
    elif args.dataset == "ImageNet":
        train_loader, test_loader, num_classes = load_dataset(args=args, dataset=args.dataset,
                                                              batch_size=args.batch_size,
                                                              dataset_path='/home0/EDA/dataset/imagenet-1k/ImageNet/data/ImageNet2012/imagenet/',
                                                              is_cuda=is_cuda)
    else:
        train_loader, test_loader, num_classes = load_dataset(args=args, dataset=args.dataset,
                                                              batch_size=args.batch_size,
                                                              dataset_path='../data', is_cuda=is_cuda)

    if args.surrogate_function == 'sigmoid':
        surrogate_function = surrogate.Sigmoid()
    elif args.surrogate_function == 'expontential':
        surrogate_function = Nature_exponential(alpha=args.thresh)
    elif args.surrogate_function == 'triangle':
        surrogate_function = surrogate.PiecewiseQuadratic()

    neuron = SpikeTrace_LIF_Neuron

    # params = {
    #     'FA': args.FA,
    #     'OS': args.OS,
    #     'bn': args.bn,
    #     'bias': args.bias,
    #     'num_classes': num_classes,
    #     'neuron': neuron,
    #     'threshold': args.thresh,
    #     'tau': args.tau,
    #     'surrogate_function': surrogate_function,
    #     'print_stats': args.print_stats,
    #     'encoding': args.encoding
    # }

    # Load spiking model
    if args.model == 'CIFAR10_VGG11_BN':
        from models.CIFAR10_VGG11 import spiking_vgg11_bn

        model = spiking_vgg11_bn(os=args.OS, neuron=neuron, num_classes=num_classes, neuron_dropout=args.neuron_dropout,
                      tau=args.tau, v_threshold = args.thresh, surrogate_function=surrogate_function, detach_reset=True)

    elif args.model == 'CIFAR10_ResNet_BN':
        from models.CIFAR10_ResNet_BN import spiking_resnet18

        model = spiking_resnet18(os=args.OS, neuron=neuron, num_classes=num_classes, neuron_dropout=args.neuron_dropout,
                                 tau=args.tau,v_threshold = args.thresh,  surrogate_function=surrogate_function, detach_reset=True, c_in=3,
                                 fc_hw=1, device=device)
    elif args.model == 'CIFAR100_ResNet_BN':
        from models.CIFAR100_ResNet_BN import spiking_resnet18

        model = spiking_resnet18(os=args.OS, neuron=neuron, num_classes=num_classes, neuron_dropout=args.neuron_dropout,
                                 tau=args.tau, v_threshold = args.thresh, surrogate_function=surrogate_function, detach_reset=True, c_in=3,
                                 fc_hw=1, device=device)
    # elif args.model == 'ImageNet_NF_ResNet':
    #     from models.ImageNet_NF_ResNet import SpikingNFResNet
    #
    #     model = SpikingNFResNet(params=params)
    elif args.model == 'DVSCIFAR10_VGG11_BN':
        from models.DVSCIFAR10_VGG11 import spiking_vgg11_bn

        model = spiking_vgg11_bn(os=args.OS, neuron=neuron, num_classes=num_classes, neuron_dropout=args.neuron_dropout,
                                 tau=args.tau, v_threshold = args.thresh, surrogate_function=surrogate_function, detach_reset=True)
    elif args.model == 'DVSGesture_VGG11_BN':
        from models.DVSGesture_VGG11 import spiking_vgg11_bn

        model = spiking_vgg11_bn(os=args.OS, neuron=neuron, num_classes=num_classes, neuron_dropout=args.neuron_dropout,
                                 tau=args.tau, v_threshold = args.thresh, surrogate_function=surrogate_function, detach_reset=True)
    else:
        raise Exception('No valid model is specified.')

    if is_cuda:
        # model.cuda()
        model = model.to(device)
        print(model)

    # # Define optimizer
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # define learning scheduler
    if args.scheduler == "Linear":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_milestones,
                                                         gamma=args.lr_decay_fact)
    elif args.scheduler == "Cosine":
        # iters = len(train_loader)/args.batch_size
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    start_epoch = 1
    best_acc = 0.
    if args.resume:
        # Load checkpoint.
        logging.info('==> Resuming from checkpoint..')
        # checkpoint = torch.load(os.path.join(save_path, 'checkpoint.pth.tar'))
        checkpoint = torch.load('./store/best_62.17.pth.tar')
        # checkpoint = torch.load('./store/2024-04-26_19-21-37/best.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        best_acc = checkpoint['best_accuracy']
        print(best_acc)
        # start_epoch = checkpoint['epoch']

    # Train loop
    for epoch in range(start_epoch, args.epochs + 1):
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        #
        # start.record()

        train_loss, train_acc, train_acc_list = train(args, model, epoch, device, optimizer, args.time_window)

        scheduler.step()

        # end.record()
        # torch.cuda.synchronize()
        # print(start.elapsed_time(end))

        test_acc, test_acc_list = test(args, model, device, args.time_window)

        # train_res[str(epoch)] = train_err_list
        # test_res[str(epoch)] = test_err_list

        # Save checkpoints and the best model
        is_best = test_acc > best_acc
        best_acc = max(best_acc, test_acc)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_accuracy': best_acc, },
                        is_best, save_path)
    logging.info(f'best acc :{best_acc}')
