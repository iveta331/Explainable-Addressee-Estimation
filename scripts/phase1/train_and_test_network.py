import os
from XAE.BASE.loaders import data_loader
from XAE.BASE.utils import create_paths, save3, calculate_performance
from XAE.PHASE1.configs import Config
from XAE.PHASE1.train import train_model
import wandb
import pyrallis
from XAE.PHASE1.test import test
import numpy as np
from XAE.BASE.plots import plot_confusion_matrix


@pyrallis.wrap()
def main(cfg: Config, wandb_cfg=None):
    if wandb_cfg is not None:
        cfg.update(wandb_cfg)

    print('Running on device: {}'.format(cfg.device))

    if cfg.log not in ['runs', 'total']:
        raise ValueError('Unsupported log type')

    cfg.save_models = True

    total_loss = np.zeros(cfg.num_runs)
    total_acc = np.zeros(cfg.num_runs)
    total_samples = 0
    cumulative_conf_mat = np.zeros((cfg.num_classes, cfg.num_classes, cfg.num_runs))
    for slot_test in range(10):  # iterating through the test list (conversation indices <0, 9>)
        cfg.slot_test = slot_test
        dataloader = data_loader(data_dir=cfg.data_dir, slot_test=cfg.slot_test, slot_eval=cfg.slot_eval,
                                 n_seq=cfg.n_seq, norm=cfg.normalisation, crop=cfg.crop, rot=cfg.rot,
                                 ang=cfg.ang, jit=cfg.jit, bri=cfg.bri, con=cfg.con, sat=cfg.sat, hue=cfg.hue,
                                 ker=cfg.ker, sig=cfg.sig)
        if cfg.save_models:
            save_path = os.path.join(cfg.save_dir, 'config.yaml')
            pyrallis.dump(cfg, open(save_path, 'w'))

        for run in range(cfg.num_runs):
            cfg.run = run
            cfg.build_nets()
            if cfg.save_models:
                create_paths(cfg)

            performances = train_model(cfg.device, cfg.slot_eval, cfg.num_epochs, cfg.log, cfg.net1, cfg.net2, cfg.net3,
                                       dataloader, cfg.save_models, cfg.plot_dir, cfg.model_name)

            if cfg.save_models:
                save3(cfg, performances, cfg.net1, cfg.net2, cfg.net3)

            test_loss, test_acc, confusion_matrix = test(cfg.device, cfg.class_names, cfg.model_name, cfg.plot_dir,
                                                         dataloader, cfg.net1, cfg.net2, cfg.net3, cfg.save_models)
            cumulative_conf_mat[:, :, run] += confusion_matrix

            total_loss[run] += test_loss * len(dataloader['test'].dataset)
            total_acc[run] += test_acc * len(dataloader['test'].dataset)
        total_samples += len(dataloader['test'].dataset)

    wandb.log({'final_weighted_test_loss': total_loss / total_samples,
               'final_weighted_test_acc': total_acc / total_samples})
    print('Confusion matrix:\n', cumulative_conf_mat)

    if cfg.save_models:
        for run in range(cfg.num_runs):
            performance_test = calculate_performance(cumulative_conf_mat[:, :, run], cfg.class_names,
                                                     f'cumulative_stats_run_{run}')
            plot_confusion_matrix(performance_test, cumulative_conf_mat[:, :, run], cfg.class_names, cfg.run_dir,
                                  f'cumulative_stats_run_{run}')


if __name__ == "__main__":
    wandb.init(project='X-AE')
    main()
