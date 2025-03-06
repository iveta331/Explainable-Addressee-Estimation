import torch
from XAE.BASE.utils import AverageMeter, calculate_performance
from XAE.BASE.plots import plot_confusion_matrix
import numpy as np
import collections
from functools import partial
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def test(device, cls_names, model_name, plot_dir, dataloader, face_network, pose_network, post_fusion_attention_network,
         save, return_maps=False, merged=False):
    """
    This test function only works for attention networks with the structure: 1)face, 1)pose, 3)attention, 4)post_fusion
    """

    num_cls = len(cls_names)
    conf_mat = np.zeros((num_cls, num_cls))
    nets_list = [face_network, pose_network, post_fusion_attention_network]
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
            classes = {0: 'ROBOT', 1: 'LEFT', 2: 'RIGHT'}
            if merged:
                images = images.reshape(images.shape[0] * images.shape[1], images.shape[2], images.shape[3],
                                        images.shape[4])

            face_tensor = face_network.model(images)
            pose_tensor = pose_network.model(poses)
            mm_tensor = torch.cat((face_tensor, pose_tensor), 1)

            output, current_batch = post_fusion_attention_network.model(mm_tensor)

            prob, preds = torch.max(output, 1)

            if return_maps:
                prob = np.exp(prob.detach().cpu().numpy())  # exp due to log_softmax
                predictions = preds.detach().cpu().numpy()

                print(prob)

                print("\nPredictions:", [(classes[k], f'{j*100:2.2f} %') for j, k in zip(prob, predictions)])
                print("Labels:", [classes[k] for k in labels.detach().cpu().numpy()])
                explanations = extract_explanations(post_fusion_attention_network.model, face_tensor, pose_tensor)

                for idx, expl in enumerate(explanations):
                    if expl == '':
                        print(f'{idx + 1}) The addressee is on the {classes[predictions[idx]]} ({prob[idx] * 100:2.2f} %)')
                    else:
                        print(f'{idx + 1}) Based on the {expl} of the interaction, '
                              f'the addressee is on the {classes[predictions[idx]]} ({prob[idx] * 100:2.2f} %)')

                plot_scores(post_fusion_attention_network.model, face_tensor, pose_tensor, images)

            loss = post_fusion_attention_network.criterion(output, labels)

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


def plot_scores(model, face, pose, imagess):
    _, importances = get_output_and_importance(model, face, pose)
    plot_data_with_images_2(importances.detach().cpu().numpy(), imagess.detach().cpu().numpy().transpose(0, 1, 3, 4, 2))


def get_output_and_importance(model, face, pose):
    def save_activation(name_l, mod, inp, out_l):
        activations[name_l] = out_l

    for name, m in model.named_modules():
        m.register_forward_hook(partial(save_activation, name))

    activations = collections.defaultdict(list)
    out = model(face, pose)

    # activations = {itm: outputs for itm, outputs in activations.items()}  # probably not needed
    feature_importances = activations['softmax_att'].squeeze()

    return out, feature_importances


def plot_data(M):
    n, m = M.shape
    fig, ax = plt.subplots()
    cax = ax.matshow(M, cmap='viridis')

    for i in range(n):
        for j in range(m):
            text = ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', color='black')

    fig.colorbar(cax)
    plt.show()


def plot_data_with_images(M, images):
    n, m, _, _, _ = images.shape
    fig = plt.figure(figsize=(10, 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(n, m+1), axes_pad=0.5)

    # Plot the matrix M
    ax_matrix = grid[0]
    cax = ax_matrix.matshow(M, cmap='viridis')

    for i in range(n):
        for j in range(m):
            ax = grid[i * (m + 1) + j + 1]
            im = images[i, j].transpose(1, 2, 0)
            im *= np.array((0.2066, 0.2014, 0.1851))
            im += np.array((0.5709, 0.4786, 0.4259))
            ax.imshow(im)
            ax.axis('off')

    for i in range(n):
        for j in range(m):
            ax_matrix.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', color='black')

    fig.colorbar(cax, ax=ax_matrix)
    plt.show()


def plot_data_with_images_2(M, images):
    n, m, _, _, _ = images.shape
    images = merge_images(images)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    # Plot the matrix M (scores) on the left side
    cax = axs[0].matshow(M, cmap='viridis', vmin=0.05, vmax=0.2)
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


def extract_explanations(model, face, pose):
    _, importances = get_output_and_importance(model, face, pose)
    importances = importances.detach().cpu().numpy()
    explanations = []
    for i in range(importances.shape[0]):
        explanations.append(get_sentence(importances[i, :]))
    return explanations


def get_sentence(arr, threshold=0.03, n_window=3):
    seq_len = arr.shape[0]

    assert n_window <= seq_len

    actual = 0
    explanations = []

    while actual+n_window < seq_len:
        actual += 1
        current_weight = np.sum(arr[actual:actual+n_window])

        if current_weight > n_window/seq_len + threshold:
            center = (2 * actual + n_window) // 2
            if center <= n_window:
                explanations.append((current_weight, 'start'))
            elif center <= 2 * n_window:
                explanations.append((current_weight, 'middle'))
            else:
                explanations.append((current_weight, 'end'))

    return max(explanations)[1] if explanations else ''

