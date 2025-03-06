from XAE.BASE.loaders import data_loader
import os
from XAE.BASE.utils import create_paths, save3
from XAE.PHASE1.configs import Config
from XAE.PHASE1.train import train_model
import wandb
import pyrallis
from XAE.PHASE1.test import test


@pyrallis.wrap()
def main(cfg: Config, wandb_cfg=None):
    if wandb_cfg is not None:
        cfg.update(wandb_cfg)

    print('Running on device: {}'.format(cfg.device))

    if cfg.log not in ['runs', 'total']:
        raise ValueError('Unsupported log type')

    total_loss = 0
    total_acc = 0
    total_samples = 0
    for slot_eval in range(10):  # iterating through the eval list (conversation indices <0, 9>)
        if slot_eval == cfg.slot_test:
            continue
        cfg.slot_eval = slot_eval
        cfg.build_nets()
        if cfg.save_models:
            create_paths(cfg)
        dataloader = data_loader(data_dir=cfg.data_dir, slot_test=cfg.slot_test, slot_eval=cfg.slot_eval,
                                 n_seq=cfg.n_seq, norm=cfg.normalisation, crop=cfg.crop, rot=cfg.rot,
                                 ang=cfg.ang, jit=cfg.jit, bri=cfg.bri, con=cfg.con, sat=cfg.sat, hue=cfg.hue,
                                 ker=cfg.ker, sig=cfg.sig)
        performances = train_model(cfg.device, cfg.slot_eval, cfg.num_epochs, cfg.log, cfg.net1, cfg.net2, cfg.net3,
                                   dataloader, cfg.save_models, cfg.plot_dir, cfg.model_name)
        total_loss += performances['eval_loss'][-1] * len(dataloader['eval'].dataset)
        total_acc += performances['eval_acc'][-1] * len(dataloader['eval'].dataset)
        total_samples += len(dataloader['eval'].dataset)
        if cfg.save_models:
            save_path = os.path.join(cfg.save_dir, 'config.yaml')
            pyrallis.dump(cfg, open(save_path, 'w'))
            save3(cfg, performances, cfg.net1, cfg.net2, cfg.net3)

    if cfg.log == 'total':
        cfg.slot_eval = None
        cfg.build_nets()
        cfg.early_stopping = False
        dataloader = data_loader(data_dir=cfg.data_dir, slot_test=cfg.slot_test, slot_eval=cfg.slot_eval,
                                 n_seq=cfg.n_seq, norm=cfg.normalisation, crop=cfg.crop, rot=cfg.rot,
                                 ang=cfg.ang, jit=cfg.jit, bri=cfg.bri, con=cfg.con, sat=cfg.sat, hue=cfg.hue,
                                 ker=cfg.ker, sig=cfg.sig)
        performances = train_model(cfg.device, cfg.slot_eval, cfg.num_epochs, cfg.log, cfg.net1, cfg.net2, cfg.net3,
                                   dataloader, cfg.save_models, cfg.plot_dir, cfg.model_name)
        test_loss, test_acc, confusin_matrix = test(cfg.device, cfg.class_names, cfg.model_name, cfg.plot_dir,
                                                    dataloader, cfg.net1, cfg.net2, cfg.net3, cfg.save_models)
        wandb.log({'final_eval_loss': total_loss / total_samples, 'final_eval_acc': total_acc / total_samples,
                   'test_loss': test_loss, 'test_acc': test_acc})


if __name__ == "__main__":
    wandb.init(project='X-AE')
    main()
