import os
from XAE.BASE.loaders import data_loader, data_loader_no_test
from XAE.BASE.utils import create_paths, save4, calculate_performance
from XAE.PHASE2.train_attention123 import train_model
from XAE.PHASE2.configs import ConfigAttComb
import wandb
import pyrallis
import numpy as np
from XAE.BASE.plots import plot_confusion_matrix


@pyrallis.wrap()
def main(cfg: ConfigAttComb, wandb_cfg=None):
    if wandb_cfg is not None:
        cfg.update(wandb_cfg)

    print('Running on device: {}'.format(cfg.device))

    if cfg.log not in ['runs', 'total']:
        raise ValueError('Unsupported log type')

    cfg.save_models = True
    cfg.slot_test = None

    total_loss = np.zeros(cfg.num_runs)
    total_acc = np.zeros(cfg.num_runs)
    total_samples = 0
    cumulative_conf_mat = np.zeros((cfg.num_classes, cfg.num_classes, cfg.num_runs))
    dataloader = data_loader_no_test(data_dir=cfg.data_dir, slot_test=cfg.slot_test,
                                     n_seq=cfg.n_seq, norm=cfg.normalisation, crop=cfg.crop, rot=cfg.rot,
                                     ang=cfg.ang, jit=cfg.jit, bri=cfg.bri, con=cfg.con, sat=cfg.sat, hue=cfg.hue,
                                     ker=cfg.ker, sig=cfg.sig, inc_icub=True)

    if cfg.save_models:
        save_path = os.path.join(cfg.save_dir, 'config.yaml')
        pyrallis.dump(cfg, open(save_path, 'w'))

    cfg.run = 0
    cfg.build_nets()
    if cfg.save_models:
        create_paths(cfg)

    performances = train_model(cfg.device, cfg.slot_eval, cfg.num_epochs, cfg.log, cfg.face_network,
                               cfg.pose_network, cfg.attention_network, cfg.post_fusion_attention_network,
                               dataloader, cfg.save_models, cfg.plot_dir, cfg.model_name, merged=True)

    if cfg.save_models:
        save4(cfg, performances, cfg.face_network, cfg.pose_network, cfg.attention_network,
              cfg.post_fusion_attention_network)

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
