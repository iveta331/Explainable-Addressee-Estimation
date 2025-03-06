from datetime import datetime
from XAE.BASE.models import FaceModel, PoseModel, FusionModelLSTM, FusionModelGRU, MergeModalities, FaceModelFeatures, \
    SelfAttention, RecurrentAttentionGRU
from XAE.PHASE2.models_att2 import FaceModelFeaturesAtt2, MergeModalitiesAtt2
import os
import torch
from dataclasses import dataclass
from pyrallis import field
from typing import Optional, Tuple
import importlib
import torch.nn as nn
from XAE.BASE.utils import get_optimizer, get_scheduler
import timm


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
    class_names: Tuple[str] = ('NAO', 'PLEFT', 'PRIGHT')  # fixed for reproducibility
    seq_len: int = 10  # length of sequence, fixed for reproducibility

    p = os.path.dirname(importlib.util.find_spec('XAE').submodule_search_locations[0])
    data_dir: str = os.path.join(p, 'data/dataset_slots')  # directory with dataset
    models_dir: str = os.path.join(p, 'models')  # directory where models will be saved
    experiment_name: str = datetime.now().strftime("%Y-%m-%d_%H-%M")  # name of this experiment
    num_runs: int = 5
    run: int = 0

    model_name: str = ''  # set in "create_paths"
    save_dir: str = ''  # set in "create_paths"
    plot_dir: str = ''  # set in "create_paths"

    save_models: bool = False  # whether to save trained models
    log: str = 'total'  # runs or total
    dev: str = '0'  # device index

    slot_test: int = 3  # testing slot index
    slot_eval: Optional[int] = None  # evaluation slot index

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

    fvp: bool = False
    vit: bool = False

    def __post_init__(self):
        self.build_nets()

    # The nets are not initialized in __init__  (and it is not present here) as it is automatically created by the
    # dataclass decorator and we do not want to overwrite it
    def build_nets(self):
        """ To be implemented in child classes """
        raise NotImplementedError

    def update(self, new_cfg):
        for key, value in new_cfg.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @property
    def device(self):
        return torch.device('cuda:' + self.dev if torch.cuda.is_available() else 'cpu')

    @property
    def activations(self) -> dict:
        return {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'Sigm': nn.Sigmoid}

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


@dataclass
class ConfigAtt1(Config):
    # OPTIMAL VALUES
    learning_rate1: float = 0.00009852
    learning_rate2: float = 0.02166
    learning_rate3: float = 0.001713

    # PARAMETERS EXCLUSIVE FOR THIS NETWORK
    optimizer_att: str = 'SGD'  # 'RMS' or 'Adam' or 'SGD'
    learning_rate_att: float = 0.00008522
    step_size_att: int = 40  # if scheduler is 'step', then every step_size epochs, lr is decreased
    weight_dec_att: float = 0.0  # weight decay for recurrent network
    momentum_att: float = 0.0  # momentum for recurrent network
    gamma_att: float = 0.9992
    scheduler_att: str = 'exponential'  # 'step' or 'exponential'

    # Parameters to merge modalities:
    dim_inner: int = 71     # dimensionality of
    dim_qk: int = 34
    merger_act: str = 'ReLU'     # or 'Tahn' or 'ReLU'

    def __post_init__(self):
        self.build_nets()

    def build_nets(self):
        # load the models
        self.face_model = FaceModel(self.conv_type, self.activations[self.act1], self.hid1, self.out1,
                                    self.dropout).to(self.device)
        self.pose_model = PoseModel(self.activations[self.act2], self.hid2, self.out2).to(self.device)

        self.post_fusion_model = FusionModelGRU(self.activations[self.act3], self.out1 + self.out2, self.hid3,
                                                self.out3, self.seq_len, self.n_seq, self.device, self.num_classes,
                                                self.dropout).to(self.device)

        self.attention_model = MergeModalities(self.dim_inner, self.out1, self.out2, self.dim_qk,
                                               self.activations[self.merger_act])

        # wrap the models in the Network class, set the corresponding schedulers
        self.face_network = Network(name='face_net',
                                    model=self.face_model,
                                    opt=get_optimizer(self.face_model, self.optimizer1, self.learning_rate1,
                                                      self.weight_dec1, self.momentum1),
                                    crit=nn.NLLLoss())
        self.face_network.scheduler = get_scheduler(self.face_network.optimizer, self.scheduler1, self.step_size1,
                                                    self.gamma1)

        self.pose_network = Network(name='pose_net',
                                    model=self.pose_model,
                                    opt=get_optimizer(self.pose_model, self.optimizer2, self.learning_rate2,
                                                      self.weight_dec2, self.momentum2),
                                    crit=nn.NLLLoss())
        self.pose_network.scheduler = get_scheduler(self.pose_network.optimizer, self.scheduler2, self.step_size2,
                                                    self.gamma2)

        self.attention_network = Network(name='attention_1_net',
                                         model=self.attention_model,
                                         opt=get_optimizer(self.attention_model, self.optimizer_att,
                                                           self.learning_rate_att, self.weight_dec_att,
                                                           self.momentum_att),
                                         crit=nn.NLLLoss())
        self.attention_network.scheduler = get_scheduler(self.attention_network.optimizer, self.scheduler_att,
                                                         self.step_size_att, self.gamma_att)

        self.post_fusion_network = Network(name='post_fusion_net',
                                           model=self.post_fusion_model,
                                           opt=get_optimizer(self.post_fusion_model, self.optimizer3,
                                                             self.learning_rate3, self.weight_dec3, self.momentum3),
                                           crit=nn.NLLLoss())
        self.post_fusion_network.scheduler = get_scheduler(self.post_fusion_network.optimizer, self.scheduler3,
                                                           self.step_size3, self.gamma3)


@dataclass
class ConfigAtt2(Config):
    optimizer_att: str = 'RMS'  # 'RMS' or 'Adam' or 'SGD'
    learning_rate_att: float = 0.000034
    step_size_att: int = 40  # if scheduler is 'step', then every step_size epochs, lr is decreased
    weight_dec_att: float = 0.0  # weight decay for recurrent network
    momentum_att: float = 0.0  # momentum for recurrent network
    gamma_att: float = 0.8306  # gamma for attention
    scheduler_att: str = 'exponential'  # 'step' or 'exponential'

    # Parameters to merge modalities:
    dim_inner: int = 71     # number of extra dimensions to process the attention computation (in the scoring function)
    dim_qk: int = 142    # dimension of the query and the key
    merger_act: str = 'ReLU'     # 'Sigm' or 'Tanh' or 'ReLU'
    kernel_size: int = 7

    # computation of values by linear projection (False if we want to keep the correspondences, e.g. after convolutions)
    value_transformation: bool = True

    # override global parameters
    out1: int = 64
    dev: str = '0'  # device index
    learning_rate1: float = 0.00001
    learning_rate2: float = 0.0023
    learning_rate3: float = 0.002682

    n_seq: int=4


    def __post_init__(self):
        self.build_nets()

    # The nets are not initialized in __init__  (and it is not present here) as it is automatically created by the
    # dataclass decorator and we do not want to overwrite it
    def build_nets(self):
        # Load the models
        self.face_model = FaceModelFeaturesAtt2(self.conv_type, self.activations[self.act1],
                                                self.dropout, ker=self.kernel_size).to(self.device)
        self.pose_model = PoseModel(self.activations[self.act2], self.hid2, self.out2).to(self.device)

        self.post_fusion_model = FusionModelGRU(self.activations[self.act3], self.out1, self.hid3,
                                                self.out3, self.seq_len, self.n_seq, self.device, self.num_classes,
                                                self.dropout).to(self.device)

        self.attention_model = MergeModalitiesAtt2(self.dim_inner, self.out1, self.out2, self.dim_qk,
                                                   self.activations[self.merger_act], self.value_transformation)

        # wrap the models in the Network class, set the corresponding schedulers
        self.face_network = Network(name='face_net',
                                    model=self.face_model,
                                    opt=get_optimizer(self.face_model, self.optimizer1, self.learning_rate1,
                                                      self.weight_dec1, self.momentum1),
                                    crit=nn.NLLLoss())
        self.face_network.scheduler = get_scheduler(self.face_network.optimizer, self.scheduler1, self.step_size1,
                                                    self.gamma1)

        self.pose_network = Network(name='pose_net',
                                    model=self.pose_model,
                                    opt=get_optimizer(self.pose_model, self.optimizer2, self.learning_rate2,
                                                      self.weight_dec2, self.momentum2),
                                    crit=nn.NLLLoss())
        self.pose_network.scheduler = get_scheduler(self.pose_network.optimizer, self.scheduler2, self.step_size2,
                                                    self.gamma2)

        self.attention_network = Network(name='attention_2_net',
                                         model=self.attention_model,
                                         opt=get_optimizer(self.attention_model, self.optimizer_att,
                                                           self.learning_rate_att, self.weight_dec_att,
                                                           self.momentum_att),
                                         crit=nn.NLLLoss())
        self.attention_network.scheduler = get_scheduler(self.attention_network.optimizer, self.scheduler_att,
                                                         self.step_size_att, self.gamma_att)

        self.post_fusion_network = Network(name='post_fusion_net',
                                           model=self.post_fusion_model,
                                           opt=get_optimizer(self.post_fusion_model, self.optimizer3,
                                                             self.learning_rate3, self.weight_dec3, self.momentum3),
                                           crit=nn.NLLLoss())
        self.post_fusion_network.scheduler = get_scheduler(self.post_fusion_network.optimizer, self.scheduler3,
                                                           self.step_size3, self.gamma3)


@dataclass
class ConfigAttComb(Config):
    gamma1: float = 0.70807  # (configured)
    gamma2: float = 0.96882  # (configured)
    gamma3: float = 0.92740  # (configured)
    gamma_att: float = 0.51241  # (configured)

    learning_rate1: float = 0.00003791  # lr for convolutional network on face images (configured)
    learning_rate2: float = 0.00117708  # lr for fully connected network processing pose vectors (configured)
    learning_rate3: float = 0.00006085  # lr for fully connected network processing pose vectors (configured)
    learning_rate_att: float = 0.00185  # (configured)

    n_seq: int = 16  # batch size (configured)
    num_epochs: int = 8  # number of epochs (configured)

    out1: int = 185    # (configured)
    # out2: int = 26  # (configured)
    out3: int = 20

    hid2: int = 73

    # PARAMETERS EXCLUSIVE TO THIS NETWORK
    optimizer_att: str = 'RMS'  # 'RMS' or 'Adam' or 'SGD'
    step_size_att: int = 40  # if scheduler is 'step', then every step_size epochs, lr is decreased
    weight_dec_att: float = 0.0  # weight decay for recurrent network
    momentum_att: float = 0.0  # momentum for recurrent network
    scheduler_att: str = 'exponential'  # 'step' or 'exponential'

    # PARAMETERS TO MERGE MODALITIES:
    dim_inner: int = 14  # number of extra dimensions to process the attention (in the scoring function)
    dim_qk: int = 20  # dimension of the query and the key
    dim_qk_gru: int = 20
    attention_act: str = 'Tanh'
    use_kv_embed: bool = True
    use_q_embed: bool = True
    dim_v: int = 81

    merger_act: str = 'ReLU'
    embed_dim: int = 42

    num_runs: int = 5

    @property
    def dim_face(self):
        return self.out1

    @property
    def dim_pose(self):
        return self.out2  # out2 is set to 20 but irrelevant, self.dim_pose is sent to the network but never used

    def __post_init__(self):
        self.build_nets()

    # The nets are not initialized in __init__  (and it is not present here) as it is automatically created by the
    # dataclass decorator and we do not want to overwrite it
    def build_nets(self):
        # load the models
        self.face_model = timm.models.vision_transformer.VisionTransformer(img_size=50, patch_size=4, in_chans=3,
                                                                           num_classes=self.out1,
                                                                           embed_dim=self.embed_dim, depth=6,
                                                                           num_heads=6).to(self.device)

        self.pose_model = PoseModel(self.activations[self.act2], self.hid2, self.out1).to(self.device)

        self.attention_model = (SelfAttention(self.dim_inner, self.out1, self.dim_qk, self.activations[self.merger_act])
                                .to(self.device))

        self.post_fusion_attention_model = RecurrentAttentionGRU(self.seq_len, self.n_seq, self.device,
                                                                 self.activations[self.attention_act],
                                                                 self.num_classes, self.out1, self.dim_pose,
                                                                 self.dim_qk_gru, self.dim_v, self.use_kv_embed,
                                                                 self.use_q_embed, use_pose=False).to(self.device)

        # wrap the models in the Network class, set the corresponding schedulers
        self.face_network = Network(name='face_net',
                                    model=self.face_model,
                                    opt=get_optimizer(self.face_model, self.optimizer1, self.learning_rate1,
                                                      self.weight_dec1,
                                                      self.momentum1),
                                    crit=nn.NLLLoss())
        self.face_network.scheduler = get_scheduler(self.face_network.optimizer, self.scheduler1, self.step_size1,
                                                    self.gamma1)

        self.pose_network = Network(name='pose_net',
                                    model=self.pose_model,
                                    opt=get_optimizer(self.pose_model, self.optimizer2, self.learning_rate2,
                                                      self.weight_dec2,
                                                      self.momentum2),
                                    crit=nn.NLLLoss())
        self.pose_network.scheduler = get_scheduler(self.pose_network.optimizer, self.scheduler2, self.step_size2,
                                                    self.gamma2)

        self.attention_network = Network(name='self_attention_net',
                                         model=self.attention_model,
                                         opt=get_optimizer(self.attention_model, self.optimizer_att,
                                                           self.learning_rate_att,
                                                           self.weight_dec_att, self.momentum_att),
                                         crit=nn.NLLLoss())
        self.attention_network.scheduler = get_scheduler(self.attention_network.optimizer, self.scheduler_att,
                                                         self.step_size_att, self.gamma_att)

        self.post_fusion_attention_network = Network(name='post_fusion_net',
                                           model=self.post_fusion_attention_model,
                                           opt=get_optimizer(self.post_fusion_attention_model, self.optimizer3,
                                                             self.learning_rate3,
                                                             self.weight_dec3, self.momentum3),
                                           crit=nn.NLLLoss())
        self.post_fusion_attention_network.scheduler = get_scheduler(self.post_fusion_attention_network.optimizer,
                                                                     self.scheduler3, self.step_size3, self.gamma3)


@dataclass
class ConfigAttNoGRU(Config):
    experiment_name: str = 'AttNoGRU'
    dev: str = '0'  # device index

    gamma1: float = 0.515198764538405  # (configured)
    gamma2: float = 0.882474752727352  # (configured)
    gamma3: float = 0.9665343663087594  # (configured)
    gamma_att: float = 0.835524861287599  # (configured)

    learning_rate1: float = 0.00001141263694798195  # lr for convolutional network on face images (configured)
    learning_rate2: float = 0.007395801210552163  # lr for fully connected network processing pose vectors (configured)
    learning_rate3: float = 0.0006808440783860434  # lr for fully connected network processing pose vectors (configured)
    learning_rate_att: float = 0.00001770229281745425  # (configured)

    n_seq: int = 16  # batch size (configured)
    num_epochs: int = 12  # number of epochs (configured)

    out1: int = 13    # (configured)
    # out2: int = 26  # (configured)
    out3: int = 40

    # PARAMETERS EXCLUSIVE TO THIS NETWORK
    optimizer_att: str = 'SGD'  # 'RMS' or 'Adam' or 'SGD'
    step_size_att: int = 40  # if scheduler is 'step', then every step_size epochs, lr is decreased
    weight_dec_att: float = 0.0  # weight decay for recurrent network
    momentum_att: float = 0.0  # momentum for recurrent network
    scheduler_att: str = 'exponential'  # 'step' or 'exponential'

    # PARAMETERS TO MERGE MODALITIES:
    dim_inner: int = 24  # number of extra dimensions to process the attention (in the scoring function)
    dim_qk: int = 167  # dimension of the query and the key

    merger_act: str = 'ReLU'
    embed_dim: int = 66

    num_runs: int = 5

    fvp: bool = True
    vit: bool = True

    @property
    def dim_face(self):
        return self.out1

    @property
    def dim_pose(self):
        return self.out2  # out2 is set to 20 but irrelevant, self.dim_pose is sent to the network but never used

    def __post_init__(self):
        self.build_nets()

    # The nets are not initialized in __init__  (and it is not present here) as it is automatically created by the
    # dataclass decorator and we do not want to overwrite it
    def build_nets(self):
        # load the models
        self.face_model = timm.models.vision_transformer.VisionTransformer(img_size=50, patch_size=4, in_chans=3,
                                                                           num_classes=self.out1,
                                                                           embed_dim=self.embed_dim, depth=6,
                                                                           num_heads=6).to(self.device)

        self.pose_model = PoseModel(self.activations[self.act2], self.hid2, self.out1).to(self.device)

        self.attention_model = (SelfAttention(self.dim_inner, self.out1, self.dim_qk, self.activations[self.merger_act])
                                .to(self.device))

        self.post_fusion_attention_model = FusionModelGRU(self.activations[self.act3], self.out1, self.hid3,
                                                          self.out3, self.seq_len, self.n_seq, self.device,
                                                          self.num_classes, self.dropout).to(self.device)

        # wrap the models in the Network class, set the corresponding schedulers
        self.face_network = Network(name='face_net',
                                    model=self.face_model,
                                    opt=get_optimizer(self.face_model, self.optimizer1, self.learning_rate1,
                                                      self.weight_dec1,
                                                      self.momentum1),
                                    crit=nn.NLLLoss())
        self.face_network.scheduler = get_scheduler(self.face_network.optimizer, self.scheduler1, self.step_size1,
                                                    self.gamma1)

        self.pose_network = Network(name='pose_net',
                                    model=self.pose_model,
                                    opt=get_optimizer(self.pose_model, self.optimizer2, self.learning_rate2,
                                                      self.weight_dec2,
                                                      self.momentum2),
                                    crit=nn.NLLLoss())
        self.pose_network.scheduler = get_scheduler(self.pose_network.optimizer, self.scheduler2, self.step_size2,
                                                    self.gamma2)

        self.attention_network = Network(name='self_attention_net',
                                         model=self.attention_model,
                                         opt=get_optimizer(self.attention_model, self.optimizer_att,
                                                           self.learning_rate_att,
                                                           self.weight_dec_att, self.momentum_att),
                                         crit=nn.NLLLoss())
        self.attention_network.scheduler = get_scheduler(self.attention_network.optimizer, self.scheduler_att,
                                                         self.step_size_att, self.gamma_att)

        self.post_fusion_attention_network = Network(name='post_fusion_net',
                                           model=self.post_fusion_attention_model,
                                           opt=get_optimizer(self.post_fusion_attention_model, self.optimizer3,
                                                             self.learning_rate3,
                                                             self.weight_dec3, self.momentum3),
                                           crit=nn.NLLLoss())
        self.post_fusion_attention_network.scheduler = get_scheduler(self.post_fusion_attention_network.optimizer,
                                                                     self.scheduler3, self.step_size3, self.gamma3)


@dataclass
class ConfigAttNoViT(Config):
    experiment_name: str = 'AttNoViT'
    dev: str = '0'  # device index

    gamma1: float = 0.9648575321102906  # (configured)
    gamma2: float = 0.8559861452524555  # (configured)
    gamma3: float = 0.568631393621717  # (configured)
    gamma_att: float = 0.7731060538976027  # (configured)

    learning_rate1: float = 0.0000062434569476523  # lr for convolutional network on face images (configured)
    learning_rate2: float = 0.0011578616259374792  # lr for fully connected network processing pose vectors (configured)
    learning_rate3: float = 0.001354478374500362  # lr for fully connected network processing pose vectors (configured)
    learning_rate_att: float = 0.00004674371395607875  # (configured)

    n_seq: int = 4  # batch size (configured)
    num_epochs: int = 14  # number of epochs (configured)

    out1: int = 152    # (configured)
    # out2: int = 26  # (configured)
    out3: int = 53

    # PARAMETERS EXCLUSIVE TO THIS NETWORK
    optimizer_att: str = 'SGD'  # 'RMS' or 'Adam' or 'SGD'
    step_size_att: int = 40  # if scheduler is 'step', then every step_size epochs, lr is decreased
    weight_dec_att: float = 0.0  # weight decay for recurrent network
    momentum_att: float = 0.0  # momentum for recurrent network
    scheduler_att: str = 'exponential'  # 'step' or 'exponential'

    # PARAMETERS TO MERGE MODALITIES:
    dim_inner: int = 129  # number of extra dimensions to process the attention (in the scoring function)
    dim_qk: int = 90  # dimension of the query and the key
    dim_qk_gru: int = 16
    attention_act: str = 'Tanh'
    use_kv_embed: bool = True
    use_q_embed: bool = True
    dim_v: int = 63

    merger_act: str = 'Tanh'

    num_runs: int = 5

    fvp: bool = True
    vit: bool = False

    @property
    def dim_face(self):
        return self.out1

    @property
    def dim_pose(self):
        return self.out2  # out2 is set to 20 but irrelevant, self.dim_pose is sent to the network but never used

    def __post_init__(self):
        self.build_nets()

    # The nets are not initialized in __init__  (and it is not present here) as it is automatically created by the
    # dataclass decorator and we do not want to overwrite it
    def build_nets(self):
        # load the models
        self.face_model = FaceModel(self.conv_type, self.activations[self.act1], self.hid1, self.out1,
                                    self.dropout).to(self.device)

        self.pose_model = PoseModel(self.activations[self.act2], self.hid2, self.out1).to(self.device)

        self.attention_model = (SelfAttention(self.dim_inner, self.out1, self.dim_qk, self.activations[self.merger_act])
                                .to(self.device))

        self.post_fusion_attention_model = RecurrentAttentionGRU(self.seq_len, self.n_seq, self.device,
                                                                 self.activations[self.attention_act],
                                                                 self.num_classes, self.out1, self.dim_pose,
                                                                 self.dim_qk_gru, self.dim_v, self.use_kv_embed,
                                                                 self.use_q_embed, use_pose=False).to(self.device)

        # wrap the models in the Network class, set the corresponding schedulers
        self.face_network = Network(name='face_net',
                                    model=self.face_model,
                                    opt=get_optimizer(self.face_model, self.optimizer1, self.learning_rate1,
                                                      self.weight_dec1,
                                                      self.momentum1),
                                    crit=nn.NLLLoss())
        self.face_network.scheduler = get_scheduler(self.face_network.optimizer, self.scheduler1, self.step_size1,
                                                    self.gamma1)

        self.pose_network = Network(name='pose_net',
                                    model=self.pose_model,
                                    opt=get_optimizer(self.pose_model, self.optimizer2, self.learning_rate2,
                                                      self.weight_dec2,
                                                      self.momentum2),
                                    crit=nn.NLLLoss())
        self.pose_network.scheduler = get_scheduler(self.pose_network.optimizer, self.scheduler2, self.step_size2,
                                                    self.gamma2)

        self.attention_network = Network(name='self_attention_net',
                                         model=self.attention_model,
                                         opt=get_optimizer(self.attention_model, self.optimizer_att,
                                                           self.learning_rate_att,
                                                           self.weight_dec_att, self.momentum_att),
                                         crit=nn.NLLLoss())
        self.attention_network.scheduler = get_scheduler(self.attention_network.optimizer, self.scheduler_att,
                                                         self.step_size_att, self.gamma_att)

        self.post_fusion_attention_network = Network(name='post_fusion_net',
                                           model=self.post_fusion_attention_model,
                                           opt=get_optimizer(self.post_fusion_attention_model, self.optimizer3,
                                                             self.learning_rate3,
                                                             self.weight_dec3, self.momentum3),
                                           crit=nn.NLLLoss())
        self.post_fusion_attention_network.scheduler = get_scheduler(self.post_fusion_attention_network.optimizer,
                                                                     self.scheduler3, self.step_size3, self.gamma3)


@dataclass
class ConfigAttNoFvP(Config):
    experiment_name: str = 'AttNoFvP'
    dev: str = '0'  # device index

    gamma1: float = 0.7826448220657637  # (configured)
    gamma2: float = 0.9968216314529336  # (configured)
    gamma3: float = 0.9768888736304692  # (configured)

    learning_rate1: float = 0.00001463139710565001  # lr for convolutional network on face images (configured)
    learning_rate2: float = 0.003745058952164801  # lr for fully connected network processing pose vectors (configured)
    learning_rate3: float = 0.00000975770287145608  # lr for fully connected network processing pose vectors (configured)

    n_seq: int = 16  # batch size (configured)
    num_epochs: int = 14  # number of epochs (configured)

    out1: int = 16    # (configured)
    out2: int = 14  # (configured)
    out3: int = 22

    # PARAMETERS EXCLUSIVE TO THIS NETWORK
    step_size_att: int = 40  # if scheduler is 'step', then every step_size epochs, lr is decreased
    weight_dec_att: float = 0.0  # weight decay for recurrent network
    momentum_att: float = 0.0  # momentum for recurrent network
    scheduler_att: str = 'exponential'  # 'step' or 'exponential'

    # PARAMETERS TO MERGE MODALITIES:
    dim_qk_gru: int = 35
    attention_act: str = 'Tanh'
    use_kv_embed: bool = True
    use_q_embed: bool = True
    dim_v: int = 102

    embed_dim: int = 180

    num_runs: int = 5

    fvp: bool = False
    vit: bool = True

    @property
    def dim_face(self):
        return self.out1

    @property
    def dim_pose(self):
        return self.out2  # out2 is set to 20 but irrelevant, self.dim_pose is sent to the network but never used

    def __post_init__(self):
        self.build_nets()

    # The nets are not initialized in __init__  (and it is not present here) as it is automatically created by the
    # dataclass decorator and we do not want to overwrite it
    def build_nets(self):
        # load the models
        self.face_model = timm.models.vision_transformer.VisionTransformer(img_size=50, patch_size=4, in_chans=3,
                                                                           num_classes=self.out1,
                                                                           embed_dim=self.embed_dim, depth=6,
                                                                           num_heads=6).to(self.device)

        self.pose_model = PoseModel(self.activations[self.act2], self.hid2, self.out2).to(self.device)

        self.post_fusion_attention_model = RecurrentAttentionGRU(self.seq_len, self.n_seq, self.device,
                                                                 self.activations[self.attention_act],
                                                                 self.num_classes, self.dim_face, self.dim_pose,
                                                                 self.dim_qk_gru, self.dim_v, self.use_kv_embed,
                                                                 self.use_q_embed, use_pose=True).to(self.device)

        # wrap the models in the Network class, set the corresponding schedulers
        self.face_network = Network(name='face_net',
                                    model=self.face_model,
                                    opt=get_optimizer(self.face_model, self.optimizer1, self.learning_rate1,
                                                      self.weight_dec1,
                                                      self.momentum1),
                                    crit=nn.NLLLoss())
        self.face_network.scheduler = get_scheduler(self.face_network.optimizer, self.scheduler1, self.step_size1,
                                                    self.gamma1)

        self.pose_network = Network(name='pose_net',
                                    model=self.pose_model,
                                    opt=get_optimizer(self.pose_model, self.optimizer2, self.learning_rate2,
                                                      self.weight_dec2,
                                                      self.momentum2),
                                    crit=nn.NLLLoss())
        self.pose_network.scheduler = get_scheduler(self.pose_network.optimizer, self.scheduler2, self.step_size2,
                                                    self.gamma2)

        self.post_fusion_attention_network = Network(name='post_fusion_net',
                                           model=self.post_fusion_attention_model,
                                           opt=get_optimizer(self.post_fusion_attention_model, self.optimizer3,
                                                             self.learning_rate3,
                                                             self.weight_dec3, self.momentum3),
                                           crit=nn.NLLLoss())
        self.post_fusion_attention_network.scheduler = get_scheduler(self.post_fusion_attention_network.optimizer,
                                                                     self.scheduler3, self.step_size3, self.gamma3)


@dataclass
class ConfigAttGRUonly(Config):
    experiment_name: str = 'AttGRUonly'

    gamma1: float = 0.9885493012716524  # (configured)
    gamma2: float = 0.6521010094542268  # (configured)
    gamma3: float = 0.6204113849369908  # (configured)

    learning_rate1: float = 0.00001573412377990971  # lr for convolutional network on face images (configured)
    learning_rate2: float = 0.003571488805052918  # lr for fully connected network processing pose vectors (configured)
    learning_rate3: float = 0.004611480725933471  # lr for fully connected network processing pose vectors (configured)

    n_seq: int = 16  # batch size (configured)
    num_epochs: int = 8  # number of epochs (configured)

    out1: int = 37    # (configured)
    out2: int = 23  # (configured)
    out3: int = 20

    # PARAMETERS EXCLUSIVE TO THIS NETWORK
    step_size_att: int = 40  # if scheduler is 'step', then every step_size epochs, lr is decreased
    weight_dec_att: float = 0.0  # weight decay for recurrent network
    momentum_att: float = 0.0  # momentum for recurrent network
    scheduler_att: str = 'exponential'  # 'step' or 'exponential'

    # PARAMETERS TO MERGE MODALITIES:
    dim_qk_gru: int = 62
    attention_act: str = 'Tanh'
    use_kv_embed: bool = True
    use_q_embed: bool = True
    dim_v: int = 53

    num_runs: int = 5

    fvp: bool = False
    vit: bool = False

    @property
    def dim_face(self):
        return self.out1

    @property
    def dim_pose(self):
        return self.out2  # out2 is set to 20 but irrelevant, self.dim_pose is sent to the network but never used

    def __post_init__(self):
        self.build_nets()

    # The nets are not initialized in __init__  (and it is not present here) as it is automatically created by the
    # dataclass decorator and we do not want to overwrite it
    def build_nets(self):
        # load the models
        self.face_model = FaceModel(self.conv_type, self.activations[self.act1], self.hid1, self.out1,
                                    self.dropout).to(self.device)

        self.pose_model = PoseModel(self.activations[self.act2], self.hid2, self.out2).to(self.device)

        self.post_fusion_attention_model = RecurrentAttentionGRU(self.seq_len, self.n_seq, self.device,
                                                                 self.activations[self.attention_act],
                                                                 self.num_classes, self.dim_face, self.dim_pose,
                                                                 self.dim_qk_gru, self.dim_v, self.use_kv_embed,
                                                                 self.use_q_embed, use_pose=True).to(self.device)

        # wrap the models in the Network class, set the corresponding schedulers
        self.face_network = Network(name='face_net',
                                    model=self.face_model,
                                    opt=get_optimizer(self.face_model, self.optimizer1, self.learning_rate1,
                                                      self.weight_dec1,
                                                      self.momentum1),
                                    crit=nn.NLLLoss())
        self.face_network.scheduler = get_scheduler(self.face_network.optimizer, self.scheduler1, self.step_size1,
                                                    self.gamma1)

        self.pose_network = Network(name='pose_net',
                                    model=self.pose_model,
                                    opt=get_optimizer(self.pose_model, self.optimizer2, self.learning_rate2,
                                                      self.weight_dec2,
                                                      self.momentum2),
                                    crit=nn.NLLLoss())
        self.pose_network.scheduler = get_scheduler(self.pose_network.optimizer, self.scheduler2, self.step_size2,
                                                    self.gamma2)

        self.post_fusion_attention_network = Network(name='post_fusion_net',
                                           model=self.post_fusion_attention_model,
                                           opt=get_optimizer(self.post_fusion_attention_model, self.optimizer3,
                                                             self.learning_rate3,
                                                             self.weight_dec3, self.momentum3),
                                           crit=nn.NLLLoss())
        self.post_fusion_attention_network.scheduler = get_scheduler(self.post_fusion_attention_network.optimizer,
                                                                     self.scheduler3, self.step_size3, self.gamma3)


@dataclass
class ConfigAttFvPonly(Config):
    experiment_name: str = 'AttFvPonly'

    gamma1: float = 0.7040027286489734  # (configured)
    gamma2: float = 0.5655703412911935  # (configured)
    gamma3: float = 0.6870177811515157  # (configured)
    gamma_att: float = 0.6881760734314196  # (configured)

    learning_rate1: float = 0.00002451015998181543  # lr for convolutional network on face images (configured)
    learning_rate2: float = 0.002531552429262647  # lr for fully connected network processing pose vectors (configured)
    learning_rate3: float = 0.0015692335670628003  # lr for fully connected network processing pose vectors (configured)
    learning_rate_att: float = 0.00009132820980623058  # (configured)

    n_seq: int = 32  # batch size (configured)
    num_epochs: int = 14  # number of epochs (configured)

    out1: int = 158    # (configured)
    # out2: int = 26  # (configured)
    out3: int = 58

    # PARAMETERS EXCLUSIVE TO THIS NETWORK
    optimizer_att: str = 'Adam'  # 'RMS' or 'Adam' or 'SGD'
    step_size_att: int = 40  # if scheduler is 'step', then every step_size epochs, lr is decreased
    weight_dec_att: float = 0.0  # weight decay for recurrent network
    momentum_att: float = 0.0  # momentum for recurrent network
    scheduler_att: str = 'exponential'  # 'step' or 'exponential'

    # PARAMETERS TO MERGE MODALITIES:
    dim_inner: int = 116  # number of extra dimensions to process the attention (in the scoring function)
    dim_qk: int = 83  # dimension of the query and the key

    merger_act: str = 'ReLU'

    num_runs: int = 5

    fvp: bool = True
    vit: bool = False

    @property
    def dim_face(self):
        return self.out1

    @property
    def dim_pose(self):
        return self.out2  # out2 is set to 20 but irrelevant, self.dim_pose is sent to the network but never used

    def __post_init__(self):
        self.build_nets()

    # The nets are not initialized in __init__  (and it is not present here) as it is automatically created by the
    # dataclass decorator and we do not want to overwrite it
    def build_nets(self):
        # load the models
        self.face_model = FaceModel(self.conv_type, self.activations[self.act1], self.hid1, self.out1,
                                    self.dropout).to(self.device)

        self.pose_model = PoseModel(self.activations[self.act2], self.hid2, self.out1).to(self.device)

        self.attention_model = (SelfAttention(self.dim_inner, self.out1, self.dim_qk, self.activations[self.merger_act])
                                .to(self.device))

        self.post_fusion_attention_model = FusionModelGRU(self.activations[self.act3], self.out1, self.hid3,
                                                          self.out3, self.seq_len, self.n_seq, self.device,
                                                          self.num_classes, self.dropout).to(self.device)

        # wrap the models in the Network class, set the corresponding schedulers
        self.face_network = Network(name='face_net',
                                    model=self.face_model,
                                    opt=get_optimizer(self.face_model, self.optimizer1, self.learning_rate1,
                                                      self.weight_dec1,
                                                      self.momentum1),
                                    crit=nn.NLLLoss())
        self.face_network.scheduler = get_scheduler(self.face_network.optimizer, self.scheduler1, self.step_size1,
                                                    self.gamma1)

        self.pose_network = Network(name='pose_net',
                                    model=self.pose_model,
                                    opt=get_optimizer(self.pose_model, self.optimizer2, self.learning_rate2,
                                                      self.weight_dec2,
                                                      self.momentum2),
                                    crit=nn.NLLLoss())
        self.pose_network.scheduler = get_scheduler(self.pose_network.optimizer, self.scheduler2, self.step_size2,
                                                    self.gamma2)

        self.attention_network = Network(name='self_attention_net',
                                         model=self.attention_model,
                                         opt=get_optimizer(self.attention_model, self.optimizer_att,
                                                           self.learning_rate_att,
                                                           self.weight_dec_att, self.momentum_att),
                                         crit=nn.NLLLoss())
        self.attention_network.scheduler = get_scheduler(self.attention_network.optimizer, self.scheduler_att,
                                                         self.step_size_att, self.gamma_att)

        self.post_fusion_attention_network = Network(name='post_fusion_net',
                                           model=self.post_fusion_attention_model,
                                           opt=get_optimizer(self.post_fusion_attention_model, self.optimizer3,
                                                             self.learning_rate3,
                                                             self.weight_dec3, self.momentum3),
                                           crit=nn.NLLLoss())
        self.post_fusion_attention_network.scheduler = get_scheduler(self.post_fusion_attention_network.optimizer,
                                                                     self.scheduler3, self.step_size3, self.gamma3)


@dataclass
class ConfigAttViTonly(Config):
    experiment_name: str = 'AttViTonly'

    gamma1: float = 0.7903465352816105  # (configured)
    gamma2: float = 0.7781619849495032  # (configured)
    gamma3: float = 0.9117016673619333  # (configured)

    learning_rate1: float = 0.00004884681028313674  # lr for convolutional network on face images (configured)
    learning_rate2: float = 0.0022428527727894145  # lr for fully connected network processing pose vectors (configured)
    learning_rate3: float = 0.00016485855004184032  # lr for fully connected network processing pose vectors (configured)

    n_seq: int = 4  # batch size (configured)
    num_epochs: int = 10  # number of epochs (configured)

    out1: int = 75    # (configured)
    out2: int = 19  # (configured)
    out3: int = 59

    # PARAMETERS EXCLUSIVE TO THIS NETWORK
    step_size_att: int = 40  # if scheduler is 'step', then every step_size epochs, lr is decreased
    weight_dec_att: float = 0.0  # weight decay for recurrent network
    momentum_att: float = 0.0  # momentum for recurrent network
    scheduler_att: str = 'exponential'  # 'step' or 'exponential'

    # PARAMETERS TO MERGE MODALITIES:

    embed_dim: int = 30

    num_runs: int = 5

    fvp: bool = False
    vit: bool = True

    @property
    def dim_face(self):
        return self.out1

    @property
    def dim_pose(self):
        return self.out2

    def __post_init__(self):
        self.build_nets()

    # The nets are not initialized in __init__  (and it is not present here) as it is automatically created by the
    # dataclass decorator and we do not want to overwrite it
    def build_nets(self):
        # load the models
        self.face_model = timm.models.vision_transformer.VisionTransformer(img_size=50, patch_size=4, in_chans=3,
                                                                           num_classes=self.out1,
                                                                           embed_dim=self.embed_dim, depth=6,
                                                                           num_heads=6).to(self.device)

        self.pose_model = PoseModel(self.activations[self.act2], self.hid2, self.out2).to(self.device)

        self.post_fusion_attention_model = FusionModelGRU(self.activations[self.act3], self.out1 + self.out2, self.hid3,
                                                          self.out3, self.seq_len, self.n_seq, self.device,
                                                          self.num_classes, self.dropout).to(self.device)

        # wrap the models in the Network class, set the corresponding schedulers
        self.face_network = Network(name='face_net',
                                    model=self.face_model,
                                    opt=get_optimizer(self.face_model, self.optimizer1, self.learning_rate1,
                                                      self.weight_dec1,
                                                      self.momentum1),
                                    crit=nn.NLLLoss())
        self.face_network.scheduler = get_scheduler(self.face_network.optimizer, self.scheduler1, self.step_size1,
                                                    self.gamma1)

        self.pose_network = Network(name='pose_net',
                                    model=self.pose_model,
                                    opt=get_optimizer(self.pose_model, self.optimizer2, self.learning_rate2,
                                                      self.weight_dec2,
                                                      self.momentum2),
                                    crit=nn.NLLLoss())
        self.pose_network.scheduler = get_scheduler(self.pose_network.optimizer, self.scheduler2, self.step_size2,
                                                    self.gamma2)

        self.post_fusion_attention_network = Network(name='post_fusion_net',
                                           model=self.post_fusion_attention_model,
                                           opt=get_optimizer(self.post_fusion_attention_model, self.optimizer3,
                                                             self.learning_rate3,
                                                             self.weight_dec3, self.momentum3),
                                           crit=nn.NLLLoss())
        self.post_fusion_attention_network.scheduler = get_scheduler(self.post_fusion_attention_network.optimizer,
                                                                     self.scheduler3, self.step_size3, self.gamma3)

