import os
from torch.utils.data import DataLoader
from XAE.BASE.utils import get_transformations
from XAE.BASE.custom_datasets import CustomDatasetVision


def data_loader(data_dir, slot_test, slot_eval, n_seq, norm, crop, rot, ang, jit, bri, con, sat, hue, ker, sig,
                up_sample_224=False):
    data_trans_face, data_trans_pose = get_transformations(norm, crop, rot, ang, jit, bri, con, sat, hue, ker, sig,
                                                           up_sample_224=up_sample_224)

    slot_folders = [x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))]
    slot_folders.sort()
    slot_folders.remove('icub')

    test = slot_folders[slot_test]
    filename_test = f'test_slot_{test}_out.csv'

    if slot_eval is not None:
        ev = slot_folders[slot_eval]
        filename_eval = f'test_slot_{ev}_out.csv'
        slot_folders.remove(ev)
        slot_folders.remove(test)
        filenames_train = [f'test_slot_{i}_out.csv' for i in slot_folders]

        dataset = {'train': CustomDatasetVision(label_dir=[os.path.join(data_dir, filenames_train[i]) for i in
                                                           range(len(slot_folders))], root_dir=data_dir,
                                                transform_face=data_trans_face['train'],
                                                transform_pose=data_trans_pose['train'], phase='train'),
                   'eval': CustomDatasetVision(label_dir=os.path.join(data_dir, filename_eval),
                                               root_dir=data_dir, transform_face=data_trans_face['eval'],
                                               transform_pose=data_trans_pose['eval'], phase='eval'),
                   'test': CustomDatasetVision(label_dir=os.path.join(data_dir, filename_test),
                                               root_dir=data_dir, transform_face=data_trans_face['test'],
                                               transform_pose=data_trans_pose['test'], phase='test')
                   }
        dataloader = {x: DataLoader(dataset[x], batch_size=n_seq, shuffle=True) for x in ['train', 'eval', 'test']}
    else:
        slot_folders.remove(test)
        filenames_train = [f'test_slot_{i}_out.csv' for i in slot_folders]
        dataset = {'train': CustomDatasetVision(label_dir=[os.path.join(data_dir, filenames_train[i]) for i in
                                                           range(len(slot_folders))], root_dir=data_dir,
                                                transform_face=data_trans_face['train'],
                                                transform_pose=data_trans_pose['train'], phase='train'),
                   'test': CustomDatasetVision(label_dir=os.path.join(data_dir, filename_test),
                                               root_dir=data_dir, transform_face=data_trans_face['test'],
                                               transform_pose=data_trans_pose['test'], phase='test')
                   }
        dataloader = {x: DataLoader(dataset[x], batch_size=n_seq, shuffle=True) for x in ['train', 'test']}

    return dataloader


def data_loader_test(data_dir, slot_test, n_seq, norm, crop, rot, ang, jit, bri, con, sat, hue, ker, sig):
    data_trans_face, data_trans_pose = get_transformations(norm, crop, rot, ang, jit, bri, con, sat, hue, ker, sig)

    slot_folders = [x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))]
    slot_folders.sort()
    slot_folders.remove('icub')

    test = slot_folders[slot_test]
    filename_test = f'test_slot_{test}_out.csv'
    dataset = CustomDatasetVision(label_dir=os.path.join(data_dir, filename_test),
                                  root_dir=data_dir, transform_face=data_trans_face['test'],
                                  transform_pose=data_trans_pose['test'], phase='test')
    dataloader = {'test': DataLoader(dataset, batch_size=n_seq, shuffle=True)}
    return dataloader


def data_loader_no_test(data_dir, slot_test, n_seq, norm, crop, rot, ang, jit, bri, con, sat, hue, ker, sig,
                        inc_icub=False):
    data_trans_face, data_trans_pose = get_transformations(norm, crop, rot, ang, jit, bri, con, sat, hue, ker, sig)

    slot_folders = [x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))]
    slot_folders.sort()
    if not inc_icub:
        slot_folders.remove('icub')

    if slot_test is None:
        if inc_icub:
            filenames_train = [f'test_interaction_{i}_out.csv' for i in range(1, 6)]
            slot_folders.remove('icub')
            filenames_train.extend([f'test_slot_{i}_out.csv' for i in slot_folders])
        else:
            filenames_train = [f'test_slot_{i}_out.csv' for i in slot_folders]
        dataset = CustomDatasetVision(label_dir=[os.path.join(data_dir, filenames_train[i]) for i in
                                                 range(len(filenames_train))], root_dir=data_dir,
                                      transform_face=data_trans_face['train'],
                                      transform_pose=data_trans_pose['train'], phase='train')
        dataloader = {'train': DataLoader(dataset, batch_size=n_seq, shuffle=True)}
    else:
        raise ValueError(f'Expected test slot to be None, instead got {slot_test}.')

    return dataloader


def data_loader_test_icub(data_dir, slot_test, n_seq, norm, crop, rot, ang, jit, bri, con, sat, hue, ker, sig):
    data_trans_face, data_trans_pose = get_transformations(norm, crop, rot, ang, jit, bri, con, sat, hue, ker, sig)

    # slot_folders = ['iCub']
    filenames = [f'test_interaction_{i}_out.csv' for i in range(1, 6)]

    if slot_test is None:
        dataset = CustomDatasetVision(label_dir=[os.path.join(data_dir, filenames[i]) for i in range(5)],
                                      root_dir=data_dir, transform_face=data_trans_face['test'],
                                      transform_pose=data_trans_pose['test'], phase='test')
        dataloader = {'test': DataLoader(dataset, batch_size=n_seq, shuffle=False)}
    else:
        raise ValueError(f'Expected test slot to be None, instead got {slot_test}.')

    return dataloader
