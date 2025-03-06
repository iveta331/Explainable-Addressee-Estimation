import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
from XAE.BASE.utils import translate_pose, calculate_possible_shift


# Need to override __init__, __len__, __getitem__
# as per datasets requirement
class CustomDatasetVision(torch.utils.data.Dataset):
    def __init__(self, label_dir, root_dir, transform_face, transform_pose, phase):
        if isinstance(label_dir, list):
            self.data = pd.concat([pd.read_csv(label_dir[i], sep=',', index_col=False, dtype=str) for i in
                                   range(len(label_dir))], ignore_index=True)
        else:
            self.data = pd.read_csv(label_dir, sep=',', index_col=False, dtype=str)
        self.root_dir = root_dir
        self.transform_face = transform_face
        self.transform_pose = transform_pose
        self.sourceTransform = None
        self.last_sequence = 0
        self.left_max_shift, self.right_max_shift = 0, 0
        self.phase = phase

    def __len__(self):
        return len(self.data) // 10

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        slot = self.data['SLOT'][idx * 10]
        if slot == '9':
            slot = '09'

        sequence = self.data['SEQUENCE'][idx * 10]
        interval = self.data['N_INTERVAL'][idx * 10]
        face_dir = [os.path.join(self.root_dir, slot, 'face_s', self.data['FACE_SPEAKER'][i])
                    for i in range(idx * 10, (idx + 1) * 10)]
        face = []
        for di in face_dir:
            di = di.replace('npy', 'jpg')

            if self.transform_face:
                face.append(np.array(self.transform_face(io.imread(di))))
            else:
                face.append(np.array(io.imread(di)))

        face = torch.tensor(np.array(face))

        pose_dir = [os.path.join(self.root_dir, slot, 'pose_s', self.data['POSE_SPEAKER'][i])
                    for i in range(idx * 10, (idx + 1) * 10)]
        pose = []
        for di in pose_dir:
            if self.transform_pose:
                pose.append(np.squeeze(np.array(self.transform_pose(np.load(di)))))
            else:
                pose.append(np.squeeze(np.array(np.load(di))))
        pose = torch.tensor(np.array(pose))
        if slot == 'icub':
            pose = torch.permute(pose, (0, 2, 1))

        pose_path = os.path.join(self.root_dir, slot, 'pose_s')
        if self.phase == 'train':
            self.left_max_shift, self.right_max_shift = calculate_possible_shift(self.data, pose_path, sequence,
                                                                                 icub=slot == 'icub')

            pose = translate_pose(pose, self.left_max_shift, self.right_max_shift)

        label = int(float(self.data['LABEL'][idx * 10]))

        return face, pose, label, sequence, interval
