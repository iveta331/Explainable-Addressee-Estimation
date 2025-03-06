import numpy as np
import os
import random
from torchvision import transforms
import torch
from torch.optim import lr_scheduler
import yaml


def create_paths(cfg):
    if cfg.slot_eval is not None:
        cfg.model_name = 'eval_' + str(cfg.slot_eval) + '_test_' + str(cfg.slot_test)
    else:
        cfg.model_name = 'test_' + str(cfg.slot_test)

    save_dir = os.path.join(cfg.run_dir, cfg.model_name)
    if not os.path.exists(cfg.models_dir):
        os.mkdir(cfg.models_dir)
    if not os.path.exists(cfg.run_dir):
        os.mkdir(cfg.run_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plot_dir = os.path.join(save_dir, 'plots')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    cfg.save_dir = save_dir
    cfg.plot_dir = plot_dir


def get_transformations(normalisation, crop, rot, ang, jit, bri, con, sat, hue, ker, sig, up_sample_224=False):
    if normalisation == 'true_stats':
        norm = transforms.Normalize(mean=[0.5709, 0.4786, 0.4259], std=[0.2066, 0.2014, 0.1851])
    elif normalisation == 'imagenet':
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif normalisation == 'none':
        norm = transforms.Normalize(mean=0, std=1)
    else:
        raise ValueError('Unsupported normalisation type')

    train = [transforms.ToPILImage()]
    if crop != 50:
        train.append(transforms.RandomCrop(crop))
        train.append(transforms.Resize((50, 50)))
    if rot:
        train.append(transforms.RandomRotation(ang))
    if jit:
        train.append(transforms.ColorJitter(bri, con, sat, hue))
    if ker != 1:
        train.append(transforms.GaussianBlur(ker, sigma=(0.1, sig)))
    train.append(transforms.ToTensor())
    train.append(norm)

    if up_sample_224:
        train.append(transforms.Resize((224, 224)))

    eval_test = [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            norm]
    if up_sample_224:
        eval_test.append(transforms.Resize((224, 224)))

    data_transform_face = {
        'train': transforms.Compose(train),
        'eval': transforms.Compose(eval_test),
        'test': transforms.Compose(eval_test)
    }

    data_transform_pose = {
        'train': transforms.ToTensor(),
        'eval': transforms.ToTensor(),
        'test': transforms.ToTensor(),
    }

    return data_transform_face, data_transform_pose


def calculate_possible_shift(df, pose_path, sequence, icub=False):
    poses = df.loc[df['SEQUENCE'] == sequence, 'POSE_SPEAKER']
    minima = []
    maxima = []
    for pose_file in poses:
        pose_dir = os.path.join(pose_path, pose_file)
        pose = np.load(pose_dir)
        if icub: pose = pose[0]
        to_delete = []
        for k, p in enumerate(pose):
            if p[2] == 0:
                to_delete.append(k)
        pose = np.delete(pose, to_delete, 0)
        minimum = min(pose[:, 0])
        maximum = max(pose[:, 0])
        minima.append(minimum)
        maxima.append(maximum)
    left_max_shift = -1-min(minima)
    right_max_shift = 1-max(maxima)

    return left_max_shift, right_max_shift


def translate_pose(poses, left_max_shift, right_max_shift):
    shift = random.uniform(left_max_shift, right_max_shift)
    for pose in poses:
        pose[pose[:, 2] != 0, 0] += shift
    return poses


def save3(cfg, performances, net1, net2, net3):
    nets_list = [net1, net2, net3]
    for net in nets_list:
        net.set_state()
        model_dir = os.path.join(cfg.save_dir, net.name + '_model.pth')
        torch.save(net.state['model_state_dict'], model_dir)
        print('Model state saved as {}/{}.pth'.format(cfg.model_name, net.name))

        optim_dir = os.path.join(cfg.save_dir, net.name + '_optimizer.pth')
        torch.save(net.state, optim_dir)
        print('Optimizer state saved as {}/{}.pth'.format(cfg.model_name, net.name))

    dict_dir = os.path.join(cfg.save_dir, 'performances.yaml')
    yaml.dump(performances, open(dict_dir, 'w'))
    print('Performances saved in performances.yaml')

def save4(cfg, performances, net1, net2, net3, net4):
    nets_list = [net1, net2, net3, net4]
    for net in nets_list:
        net.set_state()
        model_dir = os.path.join(cfg.save_dir, net.name + '_model.pth')
        torch.save(net.state['model_state_dict'], model_dir)
        print('Model state saved as {}/{}.pth'.format(cfg.model_name, net.name))

        optim_dir = os.path.join(cfg.save_dir, net.name + '_optimizer.pth')
        torch.save(net.state, optim_dir)
        print('Optimizer state saved as {}/{}.pth'.format(cfg.model_name, net.name))

    dict_dir = os.path.join(cfg.save_dir, 'performances.yaml')
    yaml.dump(performances, open(dict_dir, 'w'))
    print('Performances saved in performances.yaml')


def calculate_performance(conf_mat, cls_names, m_name, verbose=True):
    num_cls = len(cls_names)
    total_samples = np.sum(conf_mat)

    recall = np.zeros(num_cls)
    precision = np.zeros(num_cls)

    total_acc = np.sum(conf_mat[np.eye(num_cls, dtype='bool')])

    precision[0] = conf_mat[0, 0] / np.sum(conf_mat[:, 0])
    precision[1] = conf_mat[1, 1] / np.sum(conf_mat[:, 1])
    precision[2] = conf_mat[2, 2] / np.sum(conf_mat[:, 2])

    recall[0] = conf_mat[0, 0] / np.sum(conf_mat[0, :])
    recall[1] = conf_mat[1, 1] / np.sum(conf_mat[1, :])
    recall[2] = conf_mat[2, 2] / np.sum(conf_mat[2, :])

    f1 = 2 * precision * recall / (precision + recall)

    accuracy = total_acc / total_samples * 100
    error_rate = 100 - accuracy

    performance_test = {'accuracy': accuracy, 'error_rate': error_rate, 'recall': recall,
                        'precision': precision, 'f1_score': f1}
    if verbose:
        print('RESULTS OF TEST {}'.format(m_name))
        print("the accuracy of the model is {} %".format(performance_test['accuracy']))
        print("the error rate of the model is {} %".format(performance_test['error_rate']))
        print("recall: \n{}".format(performance_test['recall']))
        print("precision: \n{}".format(performance_test['precision']))
        print("f1 score: \n{}".format(performance_test['f1_score']))

    return performance_test


# https://github.com/pranoyr/cnn-lstm/blob/7062a1214ca0dbb5ba07d8405f9fbcd133b1575e/utils.py#L52
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer(model, optim, lr, wd, mom):
    if optim == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=mom)
    elif optim == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optim == 'RMS':
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd, momentum=mom)
    else:
        raise ValueError('Unsupported optimizer type')


def get_scheduler(optimizer, scheduler_type, step, gamma):
    if scheduler_type == 'step':
        return lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
    elif scheduler_type == 'exponential':
        return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        raise ValueError('Unsupported scheduler type')
