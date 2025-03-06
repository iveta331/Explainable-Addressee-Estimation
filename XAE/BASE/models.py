import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class FaceModelFeatures(nn.Module):  # altered FaceModel so that the output elements correspond to image patches
    def __init__(self, size, act, hid, out, drop):
        super(FaceModelFeatures, self).__init__()

        if size == 'small':
            chan = [6, 8, 12, 16]
        elif size == 'medium':
            chan = [8, 12, 16, 32]
        elif size == 'large':
            chan = [8, 16, 32, 64]
        else:
            raise ValueError(f'Convolutional network of size "{size}" is not supported!')

        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=chan[0], kernel_size=5)
        self.a1 = act()
        self.conv_layer2 = nn.Conv2d(in_channels=chan[0], out_channels=chan[1], kernel_size=5)
        self.a2 = act()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.d1 = nn.Dropout(p=drop)

        self.conv_layer3 = nn.Conv2d(in_channels=chan[1], out_channels=chan[2], kernel_size=5)
        self.a3 = act()
        self.conv_layer4 = nn.Conv2d(in_channels=chan[2], out_channels=chan[3], kernel_size=5)

    def forward(self, x):
        out = torch.flatten(x, 0, 1)
        out = self.conv_layer1(out)
        out = self.a1(out)
        out = self.conv_layer2(out)
        out = self.a2(out)
        out = self.max_pool1(out)
        out = self.d1(out)

        out = self.conv_layer3(out)
        out = self.a3(out)
        out = self.conv_layer4(out)

        # out = torch.flatten(out, 1)
        return out


class MergeModalities(nn.Module):
    """
    Used in Att 1
    """
    def __init__(self, dim_inner, dim_face, dim_pose, dim_kq, act, value_transform=True):
        super(MergeModalities, self).__init__()

        self.dim_inner = dim_inner
        self.dim_face = dim_face
        self.dim_pose = dim_pose

        self.dimKQ = dim_kq

        # (the same as the input to the recurrent net)
        self.dimV = self.dim_face + self.dim_pose if value_transform else self.dim_face

        # define all necessary variables
        self.WQ = nn.Linear(self.dim_pose, self.dimKQ)
        self.WK = nn.Linear(self.dim_face, self.dimKQ)
        self.WV = nn.Linear(self.dim_face, self.dimV) if value_transform else nn.Identity()

        # initialization of parameters with default normalisation
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        wd_init = torch.rand(self.dim_inner, self.dimV) - 0.5
        wd_init *= 2 * torch.sqrt(torch.tensor(1/self.dimV))
        self.WD = nn.Parameter(wd_init, requires_grad=True)

        w1_init = torch.rand(self.dim_inner, self.dimKQ) - 0.5
        w1_init *= 2 * torch.sqrt(torch.tensor(1 / self.dimKQ))
        self.W1 = nn.Parameter(w1_init, requires_grad=True)

        w2_init = torch.rand(self.dim_inner, self.dimKQ) - 0.5
        w2_init *= 2 * torch.sqrt(torch.tensor(1 / self.dimKQ))
        self.W2 = nn.Parameter(w2_init, requires_grad=True)

        self.b = nn.Parameter(torch.randn(self.dim_inner), requires_grad=True)

        self.feature_importance = nn.Identity()

        self.act = act()

    def forward(self, face, pose):
        q = self.WQ(pose)
        k = self.WK(face)
        v = self.WV(face)

        a = self.act(self.W1 @ q.T + self.W2 @ k.T + torch.atleast_2d(self.b).T)
        e = self.WD.T @ a

        e = self.feature_importance(e)  # due to easy activation extraction
        out = v * e.T
        print(e.T.shape)

        return out


class SelfAttention(nn.Module):
    def __init__(self, dim_inner, dim_feature, dim_k, act):
        super(SelfAttention, self).__init__()

        self.dim_inner = dim_inner
        self.dim_feature = dim_feature
        self.dimK = dim_k

        # define all necessary variables
        self.WK = nn.Linear(self.dim_feature, self.dimK)

        # initialization of parameters with default normalisation
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        wd_init = torch.rand(self.dim_inner) - 0.5
        wd_init *= 2 * torch.sqrt(torch.tensor(1/self.dim_feature))
        self.WD = nn.Parameter(wd_init, requires_grad=True)

        w_init = torch.rand(self.dim_inner, self.dimK) - 0.5
        w_init *= 2 * torch.sqrt(torch.tensor(1 / self.dimK))
        self.W = nn.Parameter(w_init, requires_grad=True)

        self.b = nn.Parameter(torch.randn(self.dim_inner), requires_grad=True)

        self.feature_importance = nn.Identity()

        self.act = act()

    def forward(self, face, pose):
        k_face = self.WK(face)
        k_pose = self.WK(pose)

        e_face = self.act(k_face @ self.W.T + self.b) @ self.WD
        e_pose = self.act(k_pose @ self.W.T + self.b) @ self.WD

        m = nn.Softmax(dim=1)
        e = m(torch.cat((torch.atleast_2d(e_face).T, torch.atleast_2d(e_pose).T), dim=1))

        e = self.feature_importance(e)  # due to easy activation extraction

        out = face.T * e[:, 0] + pose.T * e[:, 1]

        return out.T


# Convolutional neural network processing the faces
class FaceModel(nn.Module):  # kernel sizes could be reconsidered
    def __init__(self, size, act, hid, out, drop):
        super(FaceModel, self).__init__()

        if size == 'small':
            chan = [6, 8, 12, 16]
        elif size == 'medium':
            chan = [8, 12, 16, 32]
        elif size == 'large':
            chan = [8, 16, 32, 64]
        else:
            raise ValueError(f'Convolutional network of size "{size}" is not supported!')

        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=chan[0], kernel_size=5)
        self.a1 = act()
        self.conv_layer2 = nn.Conv2d(in_channels=chan[0], out_channels=chan[1], kernel_size=5)
        self.a2 = act()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.d1 = nn.Dropout(p=drop)

        self.conv_layer3 = nn.Conv2d(in_channels=chan[1], out_channels=chan[2], kernel_size=5)
        self.a3 = act()
        self.conv_layer4 = nn.Conv2d(in_channels=chan[2], out_channels=chan[3], kernel_size=5)
        self.a4 = act()
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.d2 = nn.Dropout(p=drop)

        # 6 * 6 * channels
        self.fc1 = nn.Linear(36 * chan[3], hid)
        self.a5 = act()
        self.fc2 = nn.Linear(hid, out)

    def forward(self, x):
        out = torch.flatten(x, 0, 1)
        out = self.conv_layer1(out)
        out = self.a1(out)
        out = self.conv_layer2(out)
        out = self.a2(out)
        out = self.max_pool1(out)
        out = self.d1(out)

        out = self.conv_layer3(out)
        out = self.a3(out)
        out = self.conv_layer4(out)
        out = self.a4(out)
        out = self.max_pool2(out)
        out = self.d2(out)

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.a5(out)
        out = self.fc2(out)

        return out


# Fully connected neural network processing the poses
class PoseModel(nn.Module):
    def __init__(self, act, hid, out):
        super(PoseModel, self).__init__()
        self.fc1 = nn.Linear(54, hid)
        self.a1 = act()
        self.fc2 = nn.Linear(hid, out)

    def forward(self, x):
        out = torch.flatten(x, 0, 1).float()
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.a1(out)
        out = self.fc2(out)

        return out


# Recurrent neural network processing the combination of output from pose network and face network
class FusionModelLSTM(nn.Module):
    def __init__(self, act, inp, hid, out, seq_len, n_seq, device, num_cls, drop):
        super(FusionModelLSTM, self).__init__()

        self.seq_len = seq_len
        self.n_seq = n_seq
        self.device = device
        self.hidden_dim = hid

        self.lstm = nn.LSTM(input_size=inp, hidden_size=hid, num_layers=1, batch_first=True)
        self.d1 = nn.Dropout(p=drop)
        self.fc1 = nn.Linear(hid, out)
        self.a1 = act()
        self.fc2 = nn.Linear(out, num_cls)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        current_batch = x.view(-1, self.seq_len, x.size(1)).size(0)

        self.hidden = (torch.zeros(1, current_batch, self.hidden_dim).to(self.device),
                       torch.zeros(1, current_batch, self.hidden_dim).to(self.device))

        lstm_out, self.hidden = self.lstm(x.view(current_batch, self.seq_len, x.size(1)), self.hidden)

        out = self.d1(lstm_out[:, -1, :])
        out = self.fc1(out)
        out = self.a1(out)
        out = self.fc2(out)
        y_pred = self.softmax(out)

        return y_pred, current_batch


class FusionModelGRU(nn.Module):
    def __init__(self, act, inp, hid, out, seq_len, n_seq, device, num_cls, drop):
        super(FusionModelGRU, self).__init__()

        self.seq_len = seq_len
        self.n_seq = n_seq
        self.device = device
        self.hidden_dim = hid

        self.gru = nn.GRU(input_size=inp, hidden_size=hid, num_layers=1, batch_first=True)
        self.d1 = nn.Dropout(p=drop)
        self.fc1 = nn.Linear(hid, out)
        self.a1 = act()
        self.fc2 = nn.Linear(out, num_cls)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        current_batch = x.view(-1, self.seq_len, x.size(1)).size(0)

        self.hidden = torch.zeros(1, current_batch, self.hidden_dim).to(self.device)

        gru_out, self.hidden = self.gru(x.view(current_batch, self.seq_len, x.size(1)), self.hidden)

        out = self.d1(gru_out[:, -1, :])
        out = self.fc1(out)
        out = self.a1(out)
        out = self.fc2(out)
        y_pred = self.softmax(out)

        return y_pred, current_batch


class RecurrentAttentionGRU(nn.Module):
    def __init__(self, seq_len, n_seq, device, act, num_cls, dim_face, dim_pose, dim_kq=32, dim_v=64, use_kv_embed=True,
                 use_q_embed=False, use_pose=True):
        super(RecurrentAttentionGRU, self).__init__()

        self.dimQK = dim_kq
        self.dimV = dim_v

        self.dim_face = dim_face
        self.dim_pose = dim_pose
        self.dim_in = self.dim_face + self.dim_pose if use_pose else self.dim_face

        self.use_KV_embedding = use_kv_embed
        self.use_Q_embedding = use_q_embed
        self.dim_gru_inp = self.dimQK if self.use_Q_embedding else self.dim_in

        self.seq_len = seq_len
        self.n_seq = n_seq
        self.device = device

        self.dim_hid = self.dim_in if not (self.use_Q_embedding and self.use_KV_embedding) else self.dimQK
        self.gru = nn.GRU(input_size=self.dim_gru_inp, hidden_size=self.dim_hid, num_layers=1, batch_first=True)

        self.WQ = nn.Linear(self.dim_in, self.dim_gru_inp) if self.use_Q_embedding else nn.Identity()
        self.WV = nn.Linear(self.dim_in, self.dimV) if self.use_KV_embedding else nn.Identity()
        self.WK = nn.Linear(self.dim_in, self.dim_hid) if self.use_KV_embedding else nn.Identity()

        self.MLP = nn.Linear(self.dimV if self.use_KV_embedding else self.dim_in, num_cls)

        self.activation = act()

        self.softmax_att = nn.Softmax(dim=2)
        self.log_softmax_out = nn.LogSoftmax(dim=1)

    def forward(self, face, pose=False, frame=None):
        current_batch = face.view(-1, self.seq_len, face.size(1)).size(0)

        if not isinstance(pose, bool):
            # The info is merged in "face"
            pose = pose.view(current_batch, self.seq_len, pose.size(1))
            face = face.view(current_batch, self.seq_len, face.size(1))

            if frame is not None:
                face = face[:, 0:frame, :]
                pose = pose[:, 0:frame, :]

            inp = torch.cat((face, pose), 2)
        else:
            inp = face.view(current_batch, self.seq_len, face.size(1))

        inp_q_ = self.WQ(inp)  # torch.Size([16, 10, 32])
        inp_q_ = self.activation(inp_q_) if self.use_Q_embedding else inp_q_
        inp_k = self.WK(inp)  # torch.Size([16, 10, 32])
        inp_v = self.WV(inp)  # torch.Size([16, 10, 64])

        hidden = (torch.randn((1, current_batch, self.dim_hid))/100).to(self.device)

        _, hidden = self.gru(inp_q_, hidden)  # shape hidden: (1, 16, 32)

        query = hidden.permute(1, 0, 2)

        kq = torch.matmul(query, inp_k.permute(0, 2, 1)) * 1 / torch.sqrt(torch.tensor(query.shape[-1]))
        kq = self.softmax_att(kq)
        result = torch.matmul(kq, inp_v)[:, 0, :]

        result = self.activation(result)
        out = self.MLP(result)
        out = self.log_softmax_out(out)

        return out, current_batch
