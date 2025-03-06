from datetime import datetime
import os
import torch
from dataclasses import dataclass
from pyrallis import field
from typing import Optional, Tuple
import importlib
from XAE.BASE.models import FaceModel, PoseModel, FusionModelLSTM, FusionModelGRU
import torch.nn as nn
from XAE.BASE.utils import get_optimizer, get_scheduler


class Network:
    def __init__(self, name, model, opt, crit):
        self.name = name
        self.model = model
        self.optimizer = opt
        self.criterion = crit
        self.scheduler = None
        self.state = {}

    def set_state(self):
        self.state['model_state_dict'] = self.model.state_dict()
        self.state['optimizer_state_dict'] = self.optimizer.state_dict()


@dataclass
class Config:
    class_names: Tuple[str] = field(default=('NAO', 'PLEFT', 'PRIGHT'))  # fixed for reproducibility
    seq_len: int = 10  # length of sequence, fixed for reproducibility

    p = os.path.dirname(importlib.util.find_spec('XAE').submodule_search_locations[0])
    data_dir: str = os.path.join(p, 'data/dataset_slots')  # directory with dataset
    models_dir: str = os.path.join(p, 'models')  # directory where models will be saved
    experiment_name: str = datetime.now().strftime("%Y-%m-%d_%H-%M")  # name of this experiment
    num_runs: int = 5
    run: int = 1

    model_name: str = ''  # set in "create_paths"
    save_dir: str = ''  # set in "create_paths"
    plot_dir: str = ''  # set in "create_paths"

    save_models: bool = False  # whether to save trained models
    log: str = 'total'  # runs or total
    dev: str = '0'  # device index

    slot_test: int = 3  # testing slot index
    slot_eval: Optional[int] = field(default=None)  # evaluation slot index

    dropout: float = 0.2  # dropout probability

    act1: str = 'Tanh'  # 'Tanh' or 'ReLU'
    act2: str = 'Tanh'  # 'Tanh' or 'ReLU'
    act3: str = 'Tanh'  # 'Tanh' or 'ReLU'

    hid1: int = 256  # size of first fully connected layer in conv network after convolution
    hid2: int = 32  # size of first fully connected layer in FC network processing pose vectors
    hid3: int = 32  # size of hidden layer in LSTM/GRU unit

    out1: int = 32  # size of second fully connected layer in conv network after convolution
    out2: int = 20  # size of second fully connected layer in FC network processing pose vectors
    out3: int = 20  # size of fully connected layer after LSTM/GRU unit

    optimizer1: str = 'RMS'  # 'RMS' or 'Adam' or 'SGD'
    optimizer2: str = 'RMS'  # 'RMS' or 'Adam' or 'SGD'
    optimizer3: str = 'Adam'  # 'RMS' or 'Adam' or 'SGD'

    learning_rate1: float = 0.00018  # learning rate for convolutional network on face images
    learning_rate2: float = 0.01  # learning rate for fully connected network processing pose vectors
    learning_rate3: float = 0.00009  # learning rate for recurrent network processing fused data streams

    step_size1: int = 40  # if scheduler is 'step', then every step_size epochs, lr is decreased
    step_size2: int = 40  # if scheduler is 'step', then every step_size epochs, lr is decreased
    step_size3: int = 40  # if scheduler is 'step', then every step_size epochs, lr is decreased

    weight_dec1: float = 0.0  # weight decay for convolutional network
    weight_dec2: float = 0.0  # weight decay for fully connected network
    weight_dec3: float = 0.0  # weight decay for recurrent network

    momentum1: float = 0.0  # momentum for convolutional network
    momentum2: float = 0.0  # momentum for fully connected network
    momentum3: float = 0.0  # momentum for recurrent network

    gamma1: float = 0.75  # gamma for convolutional network - this is the factor by which the lr is decreased
    gamma2: float = 0.9725  # gamma for fully connected network - this is the factor by which the lr is decreased
    gamma3: float = 0.5  # gamma for recurrent network - this is the factor by which the lr is decreased

    scheduler1: str = 'exponential'  # 'step' or 'exponential'
    scheduler2: str = 'exponential'  # 'step' or 'exponential'
    scheduler3: str = 'exponential'  # 'step' or 'exponential'

    conv_type: str = 'large'  # 'small or 'medium' or 'large' - controls the number of channels in conv network
    post_fusion_type: str = 'GRU'  # 'GRU or 'LSTM'

    n_seq: int = 16  # batch size
    num_epochs: int = 15  # number of epochs

    normalisation: str = 'true_stats'  # 'true_stats' or 'imagenet'

    crop: int = 50  # how big crops we want in random crop, 50 -> no crop
    rot: bool = True  # whether to use random rotation
    ang: float = 25.0  # range for rotation if rot (in degrees)
    jit: bool = True  # whether to use random jitter
    bri: float = 0.2  # range for brightness change if jit
    con: float = 0.4  # range for contrast change if jit
    sat: float = 0.45  # range for saturation change if jit
    hue: float = 0.135  # range for hue change if jit
    ker: int = 7  # kernel size in random blur, 1 -> no blur
    sig: float = 0.8  # upper bound for sigma if blur

    def __post_init__(self):
        self.build_nets()

    # The nets are not initialized in __init__  (and it is not present here) as it is automatically created by the
    # dataclass decorator and we do not want to overwrite it
    def build_nets(self):
        self.model_image = FaceModel(self.conv_type, self.activations[self.act1], self.hid1, self.out1,
                                     self.dropout).to(self.device)
        self.model_pose = PoseModel(self.activations[self.act2], self.hid2, self.out2).to(self.device)
        if self.post_fusion_type == 'LSTM':
            self.model_post_fusion = FusionModelLSTM(self.activations[self.act3], self.out1 + self.out2, self.hid3,
                                                     self.out3, self.seq_len, self.n_seq, self.device, self.num_classes,
                                                     self.dropout).to(self.device)
        elif self.post_fusion_type == 'GRU':
            self.model_post_fusion = FusionModelGRU(self.activations[self.act3], self.out1 + self.out2, self.hid3,
                                                    self.out3, self.seq_len, self.n_seq, self.device, self.num_classes,
                                                    self.dropout).to(self.device)
        else:
            raise ValueError(f'Post-fusion model type "{self.post_fusion_type}" is not supported!')

        self.net1 = Network(name='face_net',
                            model=self.model_image,
                            opt=get_optimizer(self.model_image, self.optimizer1, self.learning_rate1, self.weight_dec1,
                                              self.momentum1),
                            crit=nn.NLLLoss())
        self.net1.scheduler = get_scheduler(self.net1.optimizer, self.scheduler1, self.step_size1, self.gamma1)
        self.net2 = Network(name='pose_net',
                            model=self.model_pose,
                            opt=get_optimizer(self.model_pose, self.optimizer2, self.learning_rate2, self.weight_dec2,
                                              self.momentum2),
                            crit=nn.NLLLoss())
        self.net2.scheduler = get_scheduler(self.net2.optimizer, self.scheduler2, self.step_size2, self.gamma2)
        self.net3 = Network(name='post_fusion_net',
                            model=self.model_post_fusion,
                            opt=get_optimizer(self.model_post_fusion, self.optimizer3, self.learning_rate3,
                                              self.weight_dec3, self.momentum3),
                            crit=nn.NLLLoss())
        self.net3.scheduler = get_scheduler(self.net3.optimizer, self.scheduler3, self.step_size3, self.gamma3)

    def update(self, new_cfg):
        for key, value in new_cfg.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @property
    def device(self):
        return torch.device('cuda:' + self.dev if torch.cuda.is_available() else 'cpu')

    @property
    def activations(self) -> dict:
        return {'ReLU': nn.ReLU, 'Tanh': nn.Tanh}

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def experiment_dir(self) -> str:
        path = os.path.join(self.models_dir, self.experiment_name)
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    @property
    def run_dir(self) -> str:
        if self.num_runs == 1:
            return self.experiment_dir
        else:
            return os.path.join(self.experiment_dir, 'run_' + str(self.run))