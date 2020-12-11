import os
import time
import argparse

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from torch.utils.data import DataLoader

from unet_folder import UnetFolder

try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    print('horovod is not installed')

from unet import UNet
from utils.loss import dice_loss
from utils.meters import *
import custom_transforms

from dataset_from_pt import *

## DDP
# python -m torch.distributed.launch --nproc_per_node=1 train_3dunet.py --multiprocessing-distributed
## horovod
# horovodrun -np 4 python train_3dunet.py --use-horovod


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', default='/data/3d_data/p2', help='path to train dataset')
    parser.add_argument('--test-data', default='/data/3d_data/p7', help='path to test dataset')

    parser.add_argument("--up-sample", help="Use ConvTranspose3d layer as up_layer", action='store_true')
    parser.add_argument("--n-workers", help="The number of data loading workers (default: 4)", type=int, default=4)
    parser.add_argument('--epochs', default=400, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N',
                        help='mini-batch size (default: 4), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')

    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')

    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('-amp', '--mixed-precision', default=False, action='store_true', help='Use automatic mixed precision.')

    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    parser.add_argument('--local_rank', default=-1, type=int, help='local rank')
    
    parser.add_argument('--use-horovod', default=False, action='store_true', help='use horovod')
    parser.add_argument('--fp16-allreduce', default=False, action='store_true', help='use horovod gradient compression')
    
    parser.add_argument('--method', default='A', choices=['A', 'B', 'C'], help='choice which dataset is used')

    return parser.parse_args()


os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'


def main():
    args = parse_args()

    if args.multiprocessing_distributed and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()


    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu

        # os.environ.setdefault("NCCL_SOCKET_IFNAME", "^lo,docker")

        print(args.dist_backend, args.dist_url, args.world_size, args.rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        
    model = UNet(1, 1, 4, args.up_sample)
    
    optimizer = torch.optim.Adam(model.parameters(), args.lr) #, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20,
                                                            verbose=True, factor=0.5, min_lr=1e-8)

    if args.use_horovod:
        print('use horovod')
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
        model = model.cuda(args.gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)

        compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

        optimizer = hvd.DistributedOptimizer(optimizer,
                                             named_parameters=model.named_parameters(),
                                             compression=compression)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    elif args.distributed:
        if args.gpu is not None:
            print('use DDP')
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        print('use DP')
        model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()

    criterion = dice_loss
    cudnn.benchmark = True

    mixed_precision = args.mixed_precision
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    if args.method == 'A':
        # A
        print('not preprocess')
        train_dataset = UnetFolder(args.train_data, 224, 112, transform=custom_transforms.Compose([
                                                                            custom_transforms.ToTensor(),
                                                                            custom_transforms.Resize(112),
                                                                            custom_transforms.RandomHorizontalFlip()
                                                                        ]))
        # train_dataset = UnetFolder('./source/p2/', 224, 112, transform=Compose([ToTensor(), Resize(112), RandomHorizontalFlip()]))
        # test_dataset = UnetFolder('./source/p7/', 224, 112, transform=Compose([ToTensor(), Resize(112)]))
    elif args.method == 'B':
        # B
        # dataset is from pt, read each time
        print('already resized and croped')
        train_dataset = TensorDataset(args.train_data, transform=custom_transforms.Compose([
                                                                    custom_transforms.ToTensor(),
                                                                    custom_transforms.Resize(112),
                                                                    custom_transforms.RandomHorizontalFlip()
                                                                    ]))
        # train_dataset = TensorDataset('./before_resized/train', transform=Compose([Resize(112), RandomHorizontalFlip()]))
        # test_dataset = TensorDataset('./before_resized/train', transform=Compose([Resize(112)]))
    elif args.method == 'C':
        # C
        # dataset is from pt, already size 112, read each time
        print('already resized and croped')
        train_dataset = TensorDataset(args.train_data, transform=custom_transforms.Compose([
                                                                    custom_transforms.ToTensor(),
                                                                    custom_transforms.RandomHorizontalFlip()]))
        # train_dataset = TensorDataset('./after_resized/train', transform=custom_transforms.Compose([
        #                                                                   custom_transforms.ToTensor(),
        #                                                                   custom_transforms.RandomHorizontalFlip()]))
        # test_dataset = TensorDataset('./after_resized/train')

    if args.use_horovod:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        num_replicas=hvd.size(),
                                                                        rank=hvd.rank())
    elif args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.n_workers, pin_memory=True, sampler=train_sampler
    )
    # test_loader = DataLoader(
    #     test_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.n_workers, pin_memory=True
    # )

    for epoch in range(args.epochs):
        
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        progress = ProgressMeter(
            len(train_loader),
            batch_time, data_time, losses,
            prefix="Epoch: [{}]".format(epoch + 1))
        
        model.train()

        end = time.time()
        start = time.time()
        for i, data in enumerate(train_loader):
            images, targets = data
            
            # measure data loading time
            data_time.update(time.time() - end)
            optimizer.zero_grad()

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
            else:
                images, targets = images.cuda(), targets.cuda()
            
            batch_size = images.size(0)
            
            with torch.cuda.amp.autocast(enabled=mixed_precision):
                output = model(images)
                loss = criterion(output, targets)

            losses.update(loss.item(), batch_size)

            scaler.scale(loss).backward()
            
            if args.use_horovod:
                optimizer.synchronize()
                with optimizer.skip_synchronize():
                    # optimizer.step()
                    scaler.step(optimizer)
            else:
                scaler.step(optimizer)
            scaler.update()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)
        
        # # check loss and decay
        if scheduler:
            scheduler.step(losses.avg)
       
        # batch_time = AverageMeter('Time', ':6.3f')
        # data_time = AverageMeter('Data', ':6.3f')
        # losses = AverageMeter('Loss', ':.4e')

        # progress = ProgressMeter(
        #     len(test_loader),
        #     batch_time, data_time, losses,
        #     prefix="Epoch: [{}]".format(epoch + 1))
        
        # model.eval()

        # end = time.time()
        # start = time.time()
        # for i, data in enumerate(test_loader):
        #     images, targets = data
            
        #     # measure data loading time
        #     data_time.update(time.time() - end)

        #     if args.gpu is not None:
        #         images = images.cuda(args.gpu, non_blocking=True)
        #         targets = targets.cuda(args.gpu, non_blocking=True)
        #     else:
        #         images, targets = images.cuda(), targets.cuda()
            
        #     batch_size = images.size(0)
            
        #     with torch.cuda.amp.autocast(enabled=mixed_precision):
        #         output = model(images)
        #         loss = 1 - criterion(output, targets)
            
        #     losses.update(loss.item(), batch_size)

        #     # measure elapsed time
        #     batch_time.update(time.time() - end)
        #     end = time.time()

        #     if i % 10 == 0:
        #         if args.multiprocessing_distributed and args.rank != 0:
        #             pass
        #         else:
        #             progress.display(i)


if __name__ == '__main__':
    main()
