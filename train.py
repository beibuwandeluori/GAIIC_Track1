import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from omegaconf import OmegaConf
import argparse

from src.dataset import get_data
from src.utils import Logger, AverageMeter, get_metric
from src.model import get_model


def get_params():
    parser = argparse.ArgumentParser(description="GAIIC_Track1  @cby Training")
    parser.add_argument("--config_file", default='./logs/baseline_tag.yaml', help="config file", type=str)
    parser.add_argument("--device_id", default=0, help="Setting the GPU id", type=int)
    parser.add_argument("--k", default=0, help="the k fold", type=int)

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    return args


def eval_model(model, epoch, eval_loader, is_save=True):
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()
    eval_process = tqdm(eval_loader)
    with torch.no_grad():
        for i, data in enumerate(eval_process):
            if i > 0:
                eval_process.set_description("Valid Epoch: %d, Loss: %.4f, ACC: %.4f" %
                                             (epoch, losses.avg.item(), accs.avg.item()))
            token, label = data
            token = {k: Variable(v.cuda(device_id)) for k, v in token.items()}
            label = Variable(label.cuda(device_id))
            y_pred = model(token)

            loss = criterion(y_pred, label)
            acc = metric(y_pred, label)
            losses.update(loss.cpu(), token['input_ids'].size(0))
            accs.update(acc, token['input_ids'].size(0))

    if is_save:
        train_logger.log(phase="val", values={
            'epoch': epoch,
            'loss': format(losses.avg.item(), '.4f'),
            'acc': format(accs.avg.item(), '.4f'),
            'lr': optimizer.param_groups[0]['lr']
        })
    print("Val:\t Loss:{0:.4f} \t ACC:{1:.4f}".format(losses.avg, accs.avg))

    return accs.avg


def train_model(model, criterion, optimizer, epoch):
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    training_process = tqdm(train_loader)
    for i, data in enumerate(training_process):
        if i > 0:
            training_process.set_description(
                "Train Epoch: %d, Loss: %.4f, ACC: %.4f" % (epoch, losses.avg.item(), accs.avg.item()))

        token, label = data
        token = {k: Variable(v.cuda(device_id)) for k, v in token.items()}
        label = Variable(label.cuda(device_id))
        # Forward pass: Compute predicted y by passing x to the network
        y_pred = model(token)
        # Compute and print loss
        loss = criterion(y_pred, label)
        acc = metric(y_pred, label)
        losses.update(loss.cpu(), token['input_ids'].size(0))
        accs.update(acc, token['input_ids'].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    train_logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(losses.avg.item(), '.4f'),
        'acc': format(accs.avg.item(), '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    print("Train:\t Loss:{0:.4f} \t ACC:{1:.4f}".format(losses.avg, accs.avg))


params = get_params()
config_file = params.config_file
args = OmegaConf.load(config_file)
args.data.fold = params.k

if __name__ == '__main__':
    print(f'Training in fold {args.data.fold}')
    device_id = params.device_id
    save_per_epoch = 1
    epoch_start = 1
    num_epochs = args.train.num_epochs + 1
    model_name = args.model.model_name.split('/')[-1]
    if args.model.use_clf_head:
        model_name += '_clf'

    writeFile = f'./output/{args.name}/logs/{model_name}/fold{args.data.fold}'
    store_name = f'./output/{args.name}/weights/{model_name}/fold{args.data.fold}'

    model = get_model(args.model)
    if "load_from" in args.model:
        model.load_state_dict(torch.load(args.model.load_from, map_location="cpu"))
        print(f'Load model in {args.model.load_from}')

    model = model.cuda(device_id)
    criterion = nn.CrossEntropyLoss()
    # criterion = BCEDicedLoss(bce_weight=1.0, dice_weight=0.5)
    # criterion = LabelSmoothing(smoothing=0.05).cuda(device_id)

    (_, _, _), (train_loader, eval_loader, test_loader) = get_data(args.data)
    train_loader, eval_loader, test_loader = train_loader(), eval_loader(), test_loader()
    metric = get_metric(args.metric)

    is_train = True
    if is_train:
        if store_name and not os.path.exists(store_name):
            os.makedirs(store_name)
        train_logger = Logger(model_name=writeFile, header=['epoch', 'loss', 'acc', 'lr'])
        # optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=4e-5)
        optimizer = optim.AdamW(model.parameters(), lr=args.train.learning_rate, weight_decay=4e-5)  # 4e-5
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=1e-4)

        best_acc = 0.5 if epoch_start == 1 else eval_model(model, epoch_start - 1, eval_loader, is_save=False)
        for epoch in range(epoch_start, num_epochs):
            train_model(model, criterion, optimizer, epoch)
            if epoch % save_per_epoch == 0 or epoch == num_epochs - 1:
                acc = eval_model(model, epoch, eval_loader)
                if best_acc < acc:
                    best_acc = acc
                    torch.save(model.state_dict(), '{}/{}_acc{:.4f}.pth'.format(store_name, epoch, acc))
            print('current best acc:', best_acc)
    else:
        start = time.time()
        eval_model(model, epoch_start, test_loader, is_save=False)
        print('Total time:', time.time() - start)
