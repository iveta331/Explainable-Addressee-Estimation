import torch
import time
from XAE.BASE.utils import AverageMeter
from XAE.BASE.plots import plot_performances
import wandb


def train_model(device, slot_eval, num_epochs, log, face_network, pose_network, post_fusion_attention_network,
                dataloader, save, plot_dir, model_name, merged=False):
    """
    This train function only works for models with 3 networks
    """

    performances = {'best_eval_acc': 0, 'train_loss': [], 'train_acc': []}

    if slot_eval is not None:
        performances['eval_loss'] = []
        performances['eval_acc'] = []
    since = time.time()

    post_fusion_attention_network.model.to(device)
    nets_list = [face_network, pose_network, post_fusion_attention_network]

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
                if merged:
                    images = images.reshape(images.shape[0] * images.shape[1], images.shape[2], images.shape[3],
                                            images.shape[4])

                face_tensor = face_network.model(images)
                pose_tensor = pose_network.model(poses)
                mm_tensor = torch.cat((face_tensor, pose_tensor), 1)

                output, current_batch = post_fusion_attention_network.model(mm_tensor)

                _, preds = torch.max(output, 1)

                loss = post_fusion_attention_network.criterion(output, labels)

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
                if merged:
                    images = images.reshape(images.shape[0] * images.shape[1], images.shape[2], images.shape[3],
                                            images.shape[4])

                images = images.to(device)
                poses = poses.to(device)
                labels = labels.to(device)
                for net in nets_list:
                    net.optimizer.zero_grad()

                with torch.set_grad_enabled(False):
                    # Forward pass
                    face_tensor = face_network.model(images)
                    pose_tensor = pose_network.model(poses)
                    mm_tensor = torch.cat((face_tensor, pose_tensor), 1)

                    output, current_batch = post_fusion_attention_network.model(mm_tensor)

                    _, preds = torch.max(output, 1)

                    loss = post_fusion_attention_network.criterion(output, labels)

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
