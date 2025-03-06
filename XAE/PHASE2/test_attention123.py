import torch
from XAE.BASE.utils import AverageMeter, calculate_performance
from XAE.BASE.plots import plot_confusion_matrix
import numpy as np
import collections
from functools import partial
from matplotlib import pyplot as plt


def test(device, cls_names, model_name, plot_dir, dataloader, face_network, pose_network, attention_network,
         post_fusion_network, save, show_importances=False, merged=False):
    """
    This test function only works for attention networks with the structure: 1)face, 1)pose, 3)attention, 4)post_fusion
    """

    num_cls = len(cls_names)
    conf_mat = np.zeros((num_cls, num_cls))
    nets_list = [face_network, pose_network, attention_network, post_fusion_network]
    for net in nets_list:
        net.model.eval()
        net.model.to(device)

    losses = AverageMeter()
    accuracies = AverageMeter()

    for i, (images, poses, labels, sequence, intervals) in enumerate(dataloader['test']):
        classes = {0: 'ROBOT', 1: 'LEFT', 2: 'RIGHT'}

        images = images.to(device)
        poses = poses.to(device)
        labels = labels.to(device)
        for net in nets_list:
            net.optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            # Forward pass
            if merged:
                images = images.reshape(images.shape[0] * images.shape[1], images.shape[2], images.shape[3],
                                        images.shape[4])

            face_tensor = face_network.model(images)
            pose_tensor = pose_network.model(poses)

            mm_tensor = attention_network.model(face_tensor, pose_tensor)
            if merged:

                output, current_batch = post_fusion_network.model(mm_tensor)
            else:
                output, current_batch = post_fusion_network.model(mm_tensor)
            _, preds = torch.max(output, 1)

            if show_importances:
                print("\nPredictions:", [classes[k] for k in preds.detach().cpu().numpy()])
                print("Labels:", [classes[k] for k in labels.detach().cpu().numpy()])
                plot_pose_vs_face(attention_network.model, face_tensor, pose_tensor, images)

            loss = post_fusion_network.criterion(output, labels)

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


def plot_activation(device, cls_names, model_name, plot_dir, dataloader, net1, net2, net3, net4, save):
    num_cls = len(cls_names)
    conf_mat = np.zeros((num_cls, num_cls))
    nets_list = [net1, net2, net3, net4]
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

            # Visualisation
            # ----------------------------------------------------------------------------------------------------------
            mm_tensor, importance = get_output_and_importance(net3.model, face_tensor, pose_tensor)

            for sample_idx in range(16):
                for im_idx in [0]:
                    im = images[sample_idx, im_idx, :, :, :].squeeze()
                    im = im.permute(1, 2, 0)
                    im = im.detach().cpu().numpy()

                    im *= np.array((0.2066, 0.2014, 0.1851))
                    im += np.array((0.5709, 0.4786, 0.4259))

                    heatmap_avg = np.zeros((50, 50))
                    heatmap_pos = np.zeros((50, 50))
                    heatmap_neg = np.zeros((50, 50))
                    heatmap_abs = np.zeros((50, 50))

                    heatmap_count = np.zeros((50, 50))+1

                    for k in range(7*7):
                        x, y = np.unravel_index(k, (7, 7))

                        num_feat = 1
                        values = importance[sample_idx*10+im_idx, num_feat*k:num_feat*(k+1)].detach().cpu().numpy()
                        value_avg = np.sum(values)
                        value_pos = np.sum(values[values > 0])
                        value_neg = np.sum(values[values < 0])
                        value_abs = np.sum(np.abs(values))

                        heatmap_avg[2*x:2*x+26, 2*y:2*y+26] += value_avg
                        heatmap_pos[2*x:2*x+26, 2*y:2*y+26] += value_pos
                        heatmap_neg[2*x:2*x+26, 2*y:2*y+26] += value_neg
                        heatmap_abs[2*x:2*x+26, 2*y:2*y+26] += value_abs

                        heatmap_count[2*x:2*x+26, 2*y:2*y+26] += 1

                    fig, axs = plt.subplots(2, ncols=5)
                    axs[0][0].imshow(im)
                    axs[0][0].title.set_text('Original')
                    axs[0][0].axis('off')

                    axs[0][1].imshow(heatmap_avg / heatmap_count)
                    axs[0][1].title.set_text('Average')
                    axs[0][1].axis('off')

                    axs[0][2].imshow(heatmap_pos / heatmap_count)
                    axs[0][2].title.set_text('Norm. positive')
                    axs[0][2].axis('off')

                    axs[0][3].imshow(heatmap_neg / heatmap_count)
                    axs[0][3].title.set_text('Norm. negative')
                    axs[0][3].axis('off')

                    axs[0][4].imshow(heatmap_abs / heatmap_count)
                    axs[0][4].title.set_text('Norm. absolute')
                    axs[0][4].axis('off')

                    axs[1][0].imshow(im)
                    axs[1][0].title.set_text('Original')
                    axs[1][0].axis('off')

                    axs[1][1].imshow(heatmap_avg)
                    axs[1][1].title.set_text('Sum')
                    axs[1][1].axis('off')

                    axs[1][2].imshow(heatmap_pos)
                    axs[1][2].title.set_text('Sum positive')
                    axs[1][2].axis('off')

                    axs[1][3].imshow(heatmap_neg)
                    axs[1][3].title.set_text('Sum negative')
                    axs[1][3].axis('off')

                    axs[1][4].imshow(heatmap_abs)
                    axs[1][4].title.set_text('Absolute')
                    axs[1][4].axis('off')

                    plt.show()
            # ----------------------------------------------------------------------------------------------------------

            output, current_batch = net4.model(mm_tensor)
            _, preds = torch.max(output, 1)

            loss = net4.criterion(output, labels)

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


def get_output_and_importance(model, face, pose):
    def save_activation(name_l, mod, inp, out_l):
        activations[name_l] = out_l

    for name, m in model.named_modules():
        m.register_forward_hook(partial(save_activation, name))

    activations = collections.defaultdict(list)
    out = model(face, pose)

    # activations = {itm: outputs for itm, outputs in activations.items()}  # probably not needed
    feature_importances = activations['feature_importance']

    return out, feature_importances


def plot_pose_vs_face(model, face, pose, imagess):
    _, importances = get_output_and_importance(model, face, pose)
    current_batch = importances.view(-1, 10, importances.size(1)).size(0)
    importances = importances.view(current_batch, 10, importances.size(1))  # 4x10x2
    importances = importances.detach().cpu().numpy()
    importances = importances[:, :, 0]    # only face, 1-this is pose

    plot_data_with_images_2(importances, imagess.detach().cpu().numpy().transpose(0, 1, 3, 4, 2))


def plot_data_with_images_2(M, images):
    n, m, _, _, _ = images.shape
    images = merge_images(images)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    # Plot the matrix M (scores) on the left side
    cax = axs[0].matshow(M, cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(cax, ax=axs[0])
    axs[0].set_title('Scores')
    for i in range(n):
        for j in range(m):
            axs[0].text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', color='black')

    # Plot the images on the right side
    axs[1].set_title('Images')

    im = images
    im *= np.array((0.2066, 0.2014, 0.1851))
    im += np.array((0.5709, 0.4786, 0.4259))
    im = im.clip(min=0, max=1)

    axs[1].imshow(im)
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


def merge_images(IM):
    # Reshape the input matrix to merge images into a single big image
    merged_image = np.concatenate([np.concatenate(IM[i], axis=1) for i in range(IM.shape[0])], axis=0)
    return merged_image
