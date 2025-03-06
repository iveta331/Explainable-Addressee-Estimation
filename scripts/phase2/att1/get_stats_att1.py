from XAE.BASE.utils import create_paths, calculate_performance
from XAE.PHASE2.configs import ConfigAtt1
from XAE.BASE.loaders import data_loader_test
from XAE.PHASE2.test_attention123 import test
from XAE.BASE.plots import plot_confusion_matrix
import torch
import os
import numpy as np


cfg = ConfigAtt1()  # get default config parameters
cfg.experiment_name = 'att1_optimized'  # FIXME specify the folder name

total_acc = np.zeros(cfg.num_runs)
total_samples = 0
total_f1 = np.zeros(cfg.num_runs)
cumulative_conf_mat = np.zeros((cfg.num_classes, cfg.num_classes, cfg.num_runs))

for run in range(cfg.num_runs):  # cycle through runs
    cfg.run = run
    for test_slot in range(10):  # cycle through different test sets
        cfg.slot_test = test_slot
        cfg.build_nets()
        create_paths(cfg)

        # load models and set them to eval phase
        cfg.face_network.model.load_state_dict(torch.load(os.path.join(cfg.save_dir,
                                                                       cfg.face_network.name + '_model.pth')))
        cfg.pose_network.model.load_state_dict(torch.load(os.path.join(cfg.save_dir,
                                                                       cfg.pose_network.name + '_model.pth')))
        cfg.attention_network.model.load_state_dict(torch.load(os.path.join(cfg.save_dir,
                                                                            cfg.attention_network.name + '_model.pth')))
        cfg.attention_network.model.to(cfg.device)
        cfg.post_fusion_network.model.load_state_dict(torch.load(os.path.join(cfg.save_dir,
                                                                              cfg.post_fusion_network.name
                                                                              + '_model.pth')))

        cfg.face_network.model.eval()
        cfg.pose_network.model.eval()
        cfg.attention_network.model.eval()
        cfg.post_fusion_network.model.eval()

        # get the corresponding data_loader
        dataloader = data_loader_test(data_dir=cfg.data_dir, slot_test=cfg.slot_test, n_seq=cfg.n_seq,
                                      norm=cfg.normalisation, crop=cfg.crop, rot=cfg.rot, ang=cfg.ang, jit=cfg.jit,
                                      bri=cfg.bri, con=cfg.con, sat=cfg.sat, hue=cfg.hue, ker=cfg.ker, sig=cfg.sig)
        loss, acc, conf_mat = test(cfg.device, cfg.class_names, cfg.model_name, cfg.plot_dir, dataloader,
                                   cfg.face_network, cfg.pose_network, cfg.attention_network, cfg.post_fusion_network,
                                   save=False)

        cumulative_conf_mat[:, :, run] += conf_mat

        # get dict with keys: accuracy, error_rate, recall, precision, f1_score
        performance_test = calculate_performance(conf_mat, cfg.class_names, cfg.model_name, verbose=False)

        num_in_cls = np.sum(conf_mat, axis=1)
        # weight f1 scores according to number of samples in each class
        f1 = np.sum(performance_test['f1_score'] * num_in_cls) / np.sum(conf_mat)
        # add f1 weighted by the size of test set and averaged across runs
        total_f1[run] += f1 * len(dataloader['test'].dataset)

        total_acc[run] += acc * len(dataloader['test'].dataset)
        total_samples += len(dataloader['test'].dataset)

total_samples /= cfg.num_runs
final_weighted_accuracies = total_acc / total_samples
final_f1 = total_f1 / total_samples

print(f'final weighted test acc: {np.mean(final_weighted_accuracies)}')
print(f'final weighted f1 scores: {final_f1}')
print(f'Average f1: {np.mean(final_f1)} with std: {np.std(final_f1)}')

# compute cumulative conf matrices and plot their avg and std
avg_conf_mat = np.mean(cumulative_conf_mat, axis=2)
std_conf_mat = np.std(cumulative_conf_mat, axis=2)

plot_confusion_matrix(None, avg_conf_mat, cfg.class_names, cfg.experiment_dir, 'avg', only_mat=True)
plot_confusion_matrix(None, std_conf_mat, cfg.class_names, cfg.experiment_dir, 'std', only_mat=True)
