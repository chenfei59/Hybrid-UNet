import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
# from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
# 添加
from evaluation_metrics import evaluation_metrics
from losses import ComboLoss, AverageMeter

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    # 添加
    # testloader
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()
    # 冻结权重
    training_params = ['LK']
    for name, param in model.named_parameters():
        frozen_flag = True
        for p in training_params:
            if p in name:
                frozen_flag = False
                break
        if frozen_flag:
            param.requires_grad = False

    # 添加
    # 评价指标
    train_matric = evaluation_metrics()
    test_metric = evaluation_metrics()

    # ce_loss = CrossEntropyLoss()
    # dice_loss = DiceLoss(num_classes)

    seg_loss = ComboLoss({'dice': 0.5, 'focal': 2.0}, per_image=False).cuda()
    ce_loss = nn.CrossEntropyLoss().cuda()
    losses = AverageMeter()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0

    # 断点恢复
    if args.resume:
        iterator = tqdm(range(args.duandian + 1, max_epoch), ncols=70)
        snap_to_load = os.path.join(args.output_dir, rf'epoch_{args.duandian}.pth')
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(snap_to_load, map_location='cpu')
        model.load_state_dict(checkpoint)
    else:
        iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        losses.reset()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = model(image_batch)
            if args.evaluate_train:
                train_matric.evaluate_batch(outputs.detach(), label_batch.detach())
            if args.log_number:
                with open(os.path.join(args.output_dir, rf"log{args.log_number}.txt"), "a+") as f:
                    f.write(str(epoch_num) + ' ')
                    f.write(str(i_batch) + ' ')
                    f.write(
                        str((torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0) == 0).sum().item()) + '\t')
                    f.write(
                        str((torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0) == 1).sum().item()) + '\t')
                    f.write(
                        str((torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0) == 2).sum().item()) + '\t')
                    f.write(
                        str((torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0) == 3).sum().item()) + '\t')
                    f.write(
                        str((torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0) == 4).sum().item()) + '\t')

                    f.write(str((label_batch == 0).sum().item()) + '\t')
                    f.write(str((label_batch == 1).sum().item()) + '\t')
                    f.write(str((label_batch == 2).sum().item()) + '\t')
                    f.write(str((label_batch == 3).sum().item()) + '\t')
                    f.write(str((label_batch == 4).sum().item()) + '\n')

            loss0 = seg_loss(outputs[:, 0, ...], label_batch,index=0)
            loss1 = seg_loss(outputs[:, 1, ...], label_batch,index=1)
            loss2 = seg_loss(outputs[:, 2, ...], label_batch,index=2)
            loss3 = seg_loss(outputs[:, 3, ...], label_batch,index=3)
            loss4 = seg_loss(outputs[:, 4, ...], label_batch,index=4)

            loss5 = ce_loss(outputs, label_batch)
            loss = 0.1 * loss0 + 0.1 * loss1 + 0.3 * loss2 + 0.3 * loss3 + 0.2 * loss4 + loss5 * 11
            # loss_ce = ce_loss(outputs, label_batch[:].long())
            # loss_dice = dice_loss(outputs, label_batch, softmax=True)
            # loss = 0.4 * loss_ce + 0.6 * loss_dice
            #
            losses.update(loss.item(), image_batch.size(0))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            # logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]#索引改了
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)#索引改了
                labs = label_batch[1, ...].unsqueeze(0) * 50#索引改了
                writer.add_image('train/GroundTruth', labs, iter_num)

        # 修改
        # loss_mean = loss_total / len(trainloader)
        loss_mean =losses.avg

        # 添加
        if args.evaluate_train:
            f1, f1_avg = train_matric.get_results()
            logging.info(
                'f1: %.3f %.3f %.3f %.3f %.3f, f1_avg: %.3f %.3f %.3f %.3f %.3f' %
                (f1[0], f1[1], f1[2], f1[3], f1[4], f1_avg[0], f1_avg[1], f1_avg[2], f1_avg[3], f1_avg[4])
            )
        if args.evaluate_test:
            f1_test, f1_test_avg = test_metric.evaluate_loader(model, testloader)
            logging.info(
                'f1_test: %.3f %.3f %.3f %.3f %.3f, f1_test_avg: %.3f %.3f %.3f %.3f %.3f' %
                (f1_test[0], f1_test[1], f1_test[2], f1_test[3], f1_test[4], f1_test_avg[0], f1_test_avg[1],
                 f1_test_avg[2],
                 f1_test_avg[3], f1_test_avg[4])
            )

        logging.info('epoch %d : loss : %f, loss_mean: %f' % (epoch_num, loss.item(), loss_mean))

        save_interval = 1  # int(max_epoch/6)
        if  (epoch_num + 1) % save_interval == 0 and epoch_num > 100:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"