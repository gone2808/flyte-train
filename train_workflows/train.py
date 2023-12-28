import time
import math
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F
from flytekitplugins.kfpytorch import PyTorch, Worker
from tensorboardX import SummaryWriter
from torch import distributed as dist
from torch import nn, optim
from torchvision import datasets, transforms

from .dataset import model_2_image


def train_one_epoch(model, optimizer, data_loader, device, writer, epoch, log_interval):
    model.train()
    model.to(device)
    start_time = time.time()
    for i, data in enumerate(data_loader):

        images = list(image.to(device) for image in data[0])
        targets = [{k: v.to(device) for k, v in t.items()} for t in data[1]]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        # Print performance statistics
        if i % log_interval == 0:
            batch_time = time.time()
            speed = (i + 1) / (batch_time - start_time)
            print('[%5d] loss: %.3f, speed: %.2f' %
                  (i, loss_value, speed))
            writer.add_scalar('Training Loss', loss_value, epoch)

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            break

        optimizer.zero_grad()

        losses.backward()
        optimizer.step()


def train(model, optimizer, data_loader, device, writer, epochs, log_interval):
    model.train()
    model.to(device)

    lr_scheduler = set_up_lr_scheduler(optimizer, lr_step_size=1, lr_gamma=0.1)

#   train multiple epochs using train_one_epoch
    for epoch in tqdm(range(epochs)):
        train_one_epoch(model, optimizer, data_loader, device, writer, epoch, log_interval)
        lr_scheduler.step()
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)


def evaluate_loss(model, data_loader, device):
    # This function assumes the data loader may be shuffled, and it returns the loss in a sorted fashion
    # using knowledge of the indices that are being trained in each batch.

    # Set the model to train mode in order to get the loss, even though we're not training.
    model.train()

    loss_list = []
    indices_list = []

    assert data_loader.batch_size == 1

    start_time = time.time()
    for i, data in enumerate(data_loader):

        images = list(image.to(device) for image in data[0])
        targets = [{k: v.to(device) for k, v in t.items()} for t in data[1]]
        indices = data[2]

        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_list.append(loss_value)
        indices_list.append(indices)

        # Print performance statistics
        if i % 100 == 0:
            batch_time = time.time()
            speed = (i + 1) / (batch_time - start_time)
            print('[%5d] loss: %.3f, speed: %.2f' %
                  (i, loss_value, speed))

    loss_list = [x for _, x in sorted(zip(indices_list, loss_list))]

    return loss_list


def evaluate_iou(model, data_loader, num_classes, device='cpu', score_thresh=0.5):
    # This function removes predictions in the output and IUO calculation that are below a confidence threshold.

    # This function assumes the data loader may be shuffled, and it returns the loss in a sorted fashion
    # using knowledge of the indices that are being trained in each batch.

    # Set the model to eval mode.
    model.eval()

    ious_list = []
    boxes_list = []
    labels_list = []
    indices_list = []

    start_time = time.time()
    for i, data in enumerate(data_loader):

        images = list(image.to(device) for image in data[0])
        ground_truths = [{k: v.to(device) for k, v in t.items()} for t in data[1]]
        indices = data[2]

        model_start = time.time()
        with torch.no_grad():
            predictions = model(images)
        model_end = time.time()

        assert len(ground_truths) == len(predictions) == len(indices)  # Check if data in dataloader is consistent

        for j, pred in enumerate(predictions):

            # Ignore boxes below the confidence threshold
            thresh_inds = pred['scores'] > score_thresh
            pred_boxes = pred['boxes'][thresh_inds]
            pred_labels = pred['labels'][thresh_inds]
            pred_scores = pred['scores'][thresh_inds]

            # Find the union of prediceted and groud truth labels and iterate through it
            all_labels = np.union1d(pred_labels.to('cpu'), ground_truths[j]['labels'].to('cpu'))

            ious = np.zeros((len(all_labels)))
            for l, label in enumerate(all_labels):

                # Find the boxes corresponding to the label
                boxes_1 = pred_boxes[pred_labels == label]
                boxes_2 = ground_truths[j]['boxes'][ground_truths[j]['labels'] == label]
                iou = torchvision.ops.box_iou(boxes_1,
                                              boxes_2).cpu()  # This method returns a matrix of the IOU of each box with every other box.

                # Consider the IOU as the maximum overlap of a box with any other box. Find the max along the axis that has the most boxes.
                if 0 in iou.shape:
                    ious[l] = 0
                else:
                    if boxes_1.shape > boxes_2.shape:
                        max_iou, _ = iou.max(dim=0)
                    else:
                        max_iou, _ = iou.max(dim=1)

                    # Compute the average iou for that label
                    ious[l] = np.mean(np.array(max_iou))

            # Take the average iou for all the labels. If there are no labels, set the iou to 0.
            if len(ious) > 0:
                ious_list.append(np.mean(ious))
            else:
                ious_list.append(0)

            boxes_list.append(model_2_image(pred_boxes.cpu(), (HEIGHT, WIDTH), (
            data[3][j][0], data[3][j][1])))  # Convert the bounding box back to teh shape of the original image
            labels_list.append(np.array(pred_labels.cpu()))
            indices_list.append(indices[j])

        # Print progress
        if i % 100 == 0:
            batch_time = time.time()
            speed = (i + 1) / (batch_time - start_time)
            print('[%5d] speed: %.2f' %
                  (i, speed))

    # Sort the data based on index, just in case shuffling was used in the dataloader
    ious_list = [x for _, x in sorted(zip(indices_list, ious_list))]
    boxes_list = [x for _, x in sorted(zip(indices_list, boxes_list))]
    labels_list = [x for _, x in sorted(zip(indices_list, labels_list))]

    return ious_list, boxes_list, labels_list


def set_up_optimizer(model, lr=0.005, momentum=0.9, weight_decay=0.0005):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer


def set_up_lr_scheduler(optimizer, lr_step_size=1, lr_gamma=0.1):
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    return lr_scheduler
