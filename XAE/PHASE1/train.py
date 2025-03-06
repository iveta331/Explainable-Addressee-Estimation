import torch
import time
from XAE.BASE.utils import AverageMeter
from XAE.BASE.plots import plot_performances
import wandb


def train_model(device, slot_eval, num_epochs, log, net1, net2, net3, dataloader, save, plot_dir, model_name):

    performances = {'best_eval_acc': 0, 'train_loss': [], 'train_acc': []}

    if slot_eval is not None:
        performances['eval_loss'] = []
        performances['eval_acc'] = []
    since = time.time()

    nets_list = [net1, net2, net3]

    for epoch in range(num_epochs):
        print("\nRunning epoch no. {}".format(epoch + 1))

        if log == 'runs':
            to_log = {}
            for net in nets_list:
                to_log[net.name+'/lr'] = net.scheduler.get_last_lr()[0]

        # TRAINING

        for net in nets_list:
            net.model.train()

        losses = AverageMeter()
        accuracies = AverageMeter()

        for i, (images, poses, labels, sequence, intervals) in enumerate(dataloader['train']):
            # Move tensors to the configured device
            images = images.to(device)
            poses = poses.to(device)
            labels = labels.to(device)
            for net in nets_list:
                net.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # Forward pass
                face_tensor = net1.model(images)
                pose_tensor = net2.model(poses)
                mm_tensor = torch.cat((face_tensor, pose_tensor), 1)

                output, current_batch = net3.model(mm_tensor)
                _, preds = torch.max(output, 1)

                loss = net3.criterion(output, labels)

                # backward + optimize
                loss.backward()
                for net in nets_list:
                    net.optimizer.step()

            batch_accuracy = torch.sum(preds == labels.data) / current_batch
            losses.update(loss.item(), current_batch)
            accuracies.update(batch_accuracy.item(), current_batch)

        print('Train set ({:d} sequences): Average loss: {:.4f} Acc: {:.4f}%'.format(
            len(dataloader['train'].dataset), losses.avg, accuracies.avg * 100))
        performances['train_loss'].append(losses.avg)
        performances['train_acc'].append(accuracies.avg)

        # EVAL

        if slot_eval is not None:
            for net in nets_list:
                net.model.eval()

            losses = AverageMeter()
            accuracies = AverageMeter()

            for i, (images, poses, labels, sequence, intervals) in enumerate(dataloader['eval']):
                # Move tensors to the configured device
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

                batch_accuracy = torch.sum(preds == labels.data) / current_batch
                losses.update(loss.item(), current_batch)
                accuracies.update(batch_accuracy.item(), current_batch)

            print('Eval set no. {} ({:d} sequences): Average loss: {:.4f} Acc: {:.4f}%'.format(
                slot_eval, len(dataloader['eval'].dataset), losses.avg, accuracies.avg * 100))
            performances['eval_loss'].append(losses.avg)
            performances['eval_acc'].append(accuracies.avg)
            if accuracies.avg > performances['best_eval_acc']:
                performances['best_eval_acc'] = accuracies.avg
            if log == 'runs':
                to_log['eval_loss'] = losses.avg
                to_log['eval_acc'] = accuracies.avg

        for net in nets_list:
            net.scheduler.step()

        if log == 'runs':
            to_log['train_loss'] = performances['train_loss'][-1]
            to_log['train_acc'] = performances['train_acc'][-1]

        if log == 'runs':
            wandb.log(to_log)

        if save:
            plot_performances(epoch + 1, performances, slot_eval, plot_dir, model_name)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    if slot_eval is not None:
        print('Best eval Acc: {:4f}'.format(performances['best_eval_acc']))

    return performances
