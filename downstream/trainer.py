import sys
import math
import numpy as np
import torch
from tqdm import tqdm


def intersect_and_union(pred_label: torch.tensor, label: torch.tensor, num_classes: int):
    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=num_classes, min=0,
        max=num_classes - 1).cpu()
    area_pred_label = torch.histc(
        pred_label.float(), bins=num_classes, min=0,
        max=num_classes - 1).cpu()
    area_label = torch.histc(
        label.float(), bins=num_classes, min=0,
        max=num_classes - 1).cpu()
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def train_one_epoch(data_loader, model, optimizer, loss_func, base_lr, tb_writer, epoch, iter_num, iter_max):
    model.train()
    ce_loss = loss_func[0]
    dice_loss = loss_func[1]
    loss_list = []
    ce_loss_list = []
    dice_loss_list = []

    data_loader = tqdm(data_loader, file=sys.stdout)
    for data in data_loader:
        # forward
        image, label = data[0].to('cuda'), data[1].to('cuda')
        output = model(image)
        loss_ce = ce_loss(output, label.long())
        loss_dice = dice_loss(output, label, softmax=True)
        loss = 0.4 * loss_ce + 0.6 * loss_dice

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # adjust lr
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * iter_num / iter_max))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        iter_num = iter_num + 1

        # record
        loss_list.append(loss.item())
        ce_loss_list.append(loss_ce.item())
        dice_loss_list.append(loss_dice.item())
        data_loader.desc = f'[train epoch {epoch}] loss: {np.mean(loss_list):.3f}'

    # tensorboard
    tb_writer.add_scalar('loss', np.mean(loss_list), epoch)
    tb_writer.add_scalar('ce_loss', np.mean(ce_loss_list), epoch)
    tb_writer.add_scalar('dice_loss', np.mean(dice_loss_list), epoch)
    tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    return iter_num
