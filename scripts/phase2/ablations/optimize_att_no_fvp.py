from XAE.BASE.loaders import data_loader
import os
from XAE.BASE.utils import create_paths, save3
from XAE.PHASE2.configs import ConfigAttNoFvP
from XAE.PHASE2.train_att_3nets import train_model
import wandb
import pyrallis
from XAE.PHASE2.test_att_3nets import test


@pyrallis.wrap()
def main(cfg: ConfigAttNoFvP, wandb_cfg=None):
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

        performances = train_model(cfg.device, cfg.slot_eval, cfg.num_epochs, cfg.log, cfg.face_network,
                                   cfg.pose_network, cfg.post_fusion_attention_network,
                                   dataloader, cfg.save_models, cfg.plot_dir, cfg.model_name, merged=True)

        total_loss += performances['eval_loss'][-1] * len(dataloader['eval'].dataset)
        total_acc += performances['eval_acc'][-1] * len(dataloader['eval'].dataset)
        total_samples += len(dataloader['eval'].dataset)

        if cfg.save_models:
            save_path = os.path.join(cfg.save_dir, 'config.yaml')
            pyrallis.dump(cfg, open(save_path, 'w'))

            save3(cfg, performances, cfg.face_network, cfg.pose_network, cfg.post_fusion_attention_network)

    if cfg.log == 'total':
        cfg.slot_eval = None
        cfg.build_nets()
        cfg.early_stopping = False

        dataloader = data_loader(data_dir=cfg.data_dir, slot_test=cfg.slot_test, slot_eval=cfg.slot_eval,
                                 n_seq=cfg.n_seq, norm=cfg.normalisation, crop=cfg.crop, rot=cfg.rot,
                                 ang=cfg.ang, jit=cfg.jit, bri=cfg.bri, con=cfg.con, sat=cfg.sat, hue=cfg.hue,
                                 ker=cfg.ker, sig=cfg.sig)

        performances = train_model(cfg.device, cfg.slot_eval, cfg.num_epochs, cfg.log, cfg.face_network,
                                   cfg.pose_network, cfg.post_fusion_attention_network,
                                   dataloader, cfg.save_models, cfg.plot_dir, cfg.model_name, merged=True)

        test_loss, test_acc, confusin_matrix = test(cfg.device, cfg.class_names, cfg.model_name, cfg.plot_dir,
                                                    dataloader, cfg.face_network, cfg.pose_network,
                                                    cfg.post_fusion_attention_network, cfg.save_models, merged=True)

        wandb.log({'final_eval_loss': total_loss / total_samples, 'final_eval_acc': total_acc / total_samples,
                   'test_loss': test_loss, 'test_acc': test_acc})


if __name__ == "__main__":
    wandb.init(project='X-AE')
    main()
