import torch
from XAE.BASE.utils import AverageMeter, calculate_performance
from XAE.BASE.plots import plot_confusion_matrix
import numpy as np


def test(device, cls_names, model_name, plot_dir, dataloader, net1, net2, net3, save):
    num_cls = len(cls_names)
    conf_mat = np.zeros((num_cls, num_cls))
    nets_list = [net1, net2, net3]
    for net in nets_list:
        net.model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()

    for i, (images, poses, labels, sequence, intervals) in enumerate(dataloader['test']):
        images = images.to(device)
        poses = poses.to(device)
        labels = labels.to(device)
        for net in nets_list:
            net.optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            # Forward pass
            face_tensor = net1.model(images)
            pose_tensor = net2.model(poses)
            mm_tensor = torch.cat((face_tensor, pose_tensor), 1)

            output, current_batch = net3.model(mm_tensor)
            _, preds = torch.max(output, 1)

            loss = net3.criterion(output, labels)

        for j, l in enumerate(labels):
            conf_mat[int(labels[j]), int(preds[j])] += 1

        batch_accuracy = torch.sum(preds == labels.data) / current_batch
        losses.update(loss.item(), current_batch)
        accuracies.update(batch_accuracy.item(), current_batch)

    print('Test set ({:d} sequences): Average loss: {:.4f} Acc: {:.4f}%'.format(
        len(dataloader['test'].dataset), losses.avg, accuracies.avg * 100))

    if save:
        performance_test = calculate_performance(conf_mat, cls_names, model_name)
        plot_confusion_matrix(performance_test, conf_mat, cls_names, plot_dir, model_name)

    return losses.avg, accuracies.avg, conf_mat
