import torch
from XAE.BASE.utils import AverageMeter, calculate_performance
from XAE.BASE.plots import plot_confusion_matrix
import numpy as np
import collections
from functools import partial
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.ndimage import gaussian_filter
from torchvision import transforms

from matplotlib import colormaps
import os


def test(device, cls_names, model_name, plot_dir, dataloader, face_network, pose_network, attention_network,
         post_fusion_network, save, show_importances=False, visualize='', return_stats=False):
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

    stats = []

    for i, (images, poses, labels, sequence, intervals) in enumerate(dataloader['test']):
        classes = {0: 'ROBOT', 1: 'LEFT', 2: 'RIGHT'}

        images = images.to(device)
        poses = poses.to(device)
        labels = labels.to(device)
        for net in nets_list:
            net.optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            # Forward pass

            # image reshape due to ViT
            images_reshaped = images.reshape(images.shape[0] * images.shape[1], images.shape[2], images.shape[3],
                                    images.shape[4])

            face_tensor = face_network.model(images_reshaped)
            pose_tensor = pose_network.model(poses)

            mm_tensor = attention_network.model(face_tensor, pose_tensor)

            output, current_batch = post_fusion_network.model(mm_tensor, pose=False)
            prob, preds = torch.max(output, 1)

            if visualize == 'face_importances':
                print("\nPredictions:", [classes[k] for k in preds.detach().cpu().numpy()])
                print("Labels:", [classes[k] for k in labels.detach().cpu().numpy()])
                plot_pose_vs_face(attention_network.model, face_tensor, pose_tensor, images, stats)

                # print(len(stats))

            elif visualize == 'face_heatmap':
                orig_act, orig_tok = get_vit_activations(face_network.model, images_reshaped, batch_size=1, repeats=1,
                                                         layer=5, dim=72, num_heads=6)

                transformations = [transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])]

                transf = transforms.Compose(transformations)

                images = images_reshaped[:, :, :48, :48]
                images = transf(images)

                orig_att_data = orig_act[0]  # the index determines the layer

                heat_map = HeatMap(orig_att_data, patch_size=4, input_image=images, segmentation_mask=False)

                heat_map.visualize_map(visualize_token=True, merge_heads=True, blur_factor=2, scale_overlay=2,
                                       save=False,
                                       save_to='save_dir',
                                       save_name='')

            elif visualize == 'frame_importances':
                prob = np.exp(prob.detach().cpu().numpy())  # exp due to log_softmax
                predictions = preds.detach().cpu().numpy()

                print("\nPredictions:", [(classes[k], f'{j * 100:2.2f} %') for j, k in zip(prob, predictions)])
                print("Labels:", [classes[k] for k in labels.detach().cpu().numpy()])

                explanations, expl = extract_explanations(post_fusion_network.model, mm_tensor)
                if return_stats:
                    stats.extend(list(expl.flatten()))

                for idx, expl in enumerate(explanations):
                    if expl == '':
                        print(
                            f'{idx + 1}) The addressee is on the {classes[predictions[idx]]} ({prob[idx] * 100:2.2f} %)')
                    else:
                        print(f'{idx + 1}) Based on the {expl} of the interaction, '
                              f'the addressee is on the {classes[predictions[idx]]} ({prob[idx] * 100:2.2f} %)')

                plot_scores(post_fusion_network.model, mm_tensor, images)

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

    return losses.avg, accuracies.avg, conf_mat, stats


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

    print("IMMMM", feature_importances.shape)

    return out, feature_importances


def plot_pose_vs_face(model, face, pose, imagess, s):
    _, importances = get_output_and_importance(model, face, pose)
    current_batch = importances.view(-1, 10, importances.size(1)).size(0)
    importances = importances.view(current_batch, 10, importances.size(1))  # 4x10x2
    importances = importances.detach().cpu().numpy()
    importances = importances[:, :, 0]    # only face, 1-this is pose

    s.extend(list(importances.flatten()))
    # print(importances.shape)
    # print(imagess.shape)
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



def extract_explanations(model, face_pose):
    _, importances = get_output_and_importance_rnn(model, face_pose)
    importances = importances.detach().cpu().numpy()
    explanations = []
    if len(importances.shape) == 2:
        for i in range(importances.shape[0]):
            explanations.append(get_sentence(importances[i, :]))
    else:
        explanations.append(get_sentence(importances[:]))

    return explanations, importances

def plot_scores(model, face_pose, imagess):
    _, importances = get_output_and_importance_rnn(model, face_pose)
    plot_data_with_images_2(importances.detach().cpu().numpy(), imagess.detach().cpu().numpy().transpose(0, 1, 3, 4, 2))


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


def get_output_and_importance_rnn(model, face_pose):
    def save_activation(name_l, mod, inp, out_l):
        activations[name_l] = out_l

    for name, m in model.named_modules():
        m.register_forward_hook(partial(save_activation, name))

    activations = collections.defaultdict(list)
    out = model(face_pose, pose=False)

    # activations = {itm: outputs for itm, outputs in activations.items()}  # probably not needed
    feature_importances = activations['softmax_att'].squeeze()

    return out, feature_importances


def get_vit_activations(model, data, batch_size, repeats=1, layer=0, dim=72, num_heads=6):
    """
    Inspiration from here:
    https://gist.github.com/Tushar-N/680633ec18f5cb4b47933da7d10902af
    """
    model.eval()
    result = []
    outs = []

    def save_activation(name_l, mod, inp, out_l):
        activations[name_l] = out_l

    for name, m in model.named_modules():
        m.register_forward_hook(partial(save_activation, name))

    for k in range(repeats):
        activations = collections.defaultdict(list)

        out = model(data)

        outs.append(out.data.max(1, keepdim=True)[1])
        activations = {itm: outputs for itm, outputs in activations.items()}

        # tokens.append(cls_token)

        if isinstance(layer, int):
            q, k = activations[f'blocks.{layer}.attn.q_norm'][0], activations[f'blocks.{layer}.attn.k_norm'][0]
            scale = (dim // num_heads) ** -0.5
            q = q * scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            result.append(attn)
        elif isinstance(layer, list):
            for la in layer:
                q, k = activations[f'blocks.{la}.attn.q_norm'][0], activations[f'blocks.{la}.attn.k_norm'][0]
                scale = (dim // num_heads) ** -0.5
                q = q * scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                result.append(attn)

    return result, outs


class HeatMap:
    def __init__(self, data, patch_size, input_image, segmentation_mask):
        """
        :param data: attention activation; Shape: (n_heads, n_patches^2+1, n_patches^2+1)
        :param patch_size: patch size of the network: (int)
        :param input_image: input to the RViT(); Shape: (1, 3, n_pixels_x, n_pixels_y)
        :param segmentation_mask: segmentation mask provided for PET dataset; Shape: (1, 1, n_pixels_x, n_pixels_y)
        """

        self.data = data
        self.patch_size = patch_size
        self.segmentation_mask = np.squeeze(np.array(segmentation_mask * 255, dtype=np.int32))

        input_image = (input_image.detach().cpu().numpy())
        input_image = np.moveaxis(input_image[0], 0, 2)
        self.input_image = input_image

    def update_data(self, data):
        self.data = data

    def _calculate_heatmap(self, vis_cls_tok=False, merge_heads=False):
        data = self.data.detach().cpu().numpy()
        _num_heads = data.shape[0]

        if vis_cls_tok:
            # extracting the attention matrix for the class token
            data = data[:, 0, 1:]  # the middle index specifies the index of attention mask (0 = cls_token)
        else:
            # sum of contributions (not only for the class token)
            data = np.sum(data, axis=1)[:, 1:]

        n_patch = int(np.sqrt((data.shape[-1])))
        data = data.reshape((_num_heads, n_patch, n_patch))
        data = np.repeat(data, self.patch_size, axis=1)
        data = np.repeat(data, self.patch_size, axis=2)

        if merge_heads:
            data = np.sum(data, axis=0)
            min_val = np.min(data, axis=1, keepdims=True)
            data -= min_val
            data /= np.max(data) - np.min(data)

        else:
            min_val = np.min(data, axis=(1, 2), keepdims=True)
            max_val = np.max(data, axis=(1, 2), keepdims=True)
            data -= min_val
            data /= max_val - min_val

            data -= 0.5
            data *= 2

        return data

    def visualize_map(self, visualize_token=True, merge_heads=True, blur_factor=0, scale_overlay=2, save=True,
                      save_to='', save_name='Image'):
        attention_map = self._calculate_heatmap(merge_heads=merge_heads, vis_cls_tok=visualize_token)
        if merge_heads:
            fig, axs = plt.subplots(1, 3)
            axs[0].imshow(np.clip(self.input_image, a_min=0, a_max=1))
            axs[0].axis('off')
            blurred = gaussian_filter(np.reshape(attention_map, (attention_map.shape[0],
                                                                 attention_map.shape[1], 1)),
                                      sigma=blur_factor)
            axs[1].imshow(np.clip(blurred, a_min=0, a_max=1), cmap=colormaps['viridis'])
            axs[1].axis('off')

            blurred *= scale_overlay
            blurred = np.clip(blurred, 0, 1)
            p = axs[2].imshow(np.clip(self.input_image * blurred, a_min=0, a_max=1))
            axs[2].axis('off')
            fig.colorbar(p, ax=axs[2])

        else:
            _num_heads = attention_map.shape[0]
            fig, axs = plt.subplots(3, ncols=_num_heads)
            for i in range(_num_heads):
                axs[0][i].imshow(self.input_image)

                blurred = gaussian_filter(np.reshape(attention_map[i], (attention_map[i].shape[0],
                                                                        attention_map[i].shape[1], 1)),
                                          sigma=blur_factor)
                axs[1][i].imshow(blurred, cmap=colormaps['plasma'])

                blurred *= scale_overlay
                blurred = np.clip(blurred, 0, 1)
                axs[2][i].imshow(self.input_image * blurred)

        if save:
            plt.savefig(os.path.join(save_to, save_name + '.png'), dpi=400)
            plt.close()
        else:
            plt.suptitle('Attention map')
            plt.draw()

            plt.waitforbuttonpress(0)
            plt.close()
