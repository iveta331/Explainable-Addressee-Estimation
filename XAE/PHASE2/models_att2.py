import torch
import torch.nn as nn


class FaceModelFeaturesAtt2(nn.Module):  # altered FaceModel so that the output elements correspond to image patches
    def __init__(self, size, act, drop, ker=5):
        super(FaceModelFeaturesAtt2, self).__init__()

        kernel_size = ker

        if size == 'small':
            chan = [6, 8, 12, 16]
        elif size == 'medium':
            chan = [8, 12, 16, 32]
        elif size == 'large':
            chan = [8, 16, 32, 64]
        else:
            raise ValueError(f'Convolutional network of size "{size}" is not supported!')

        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=chan[0], kernel_size=kernel_size)
        self.a1 = act()
        self.conv_layer2 = nn.Conv2d(in_channels=chan[0], out_channels=chan[1], kernel_size=kernel_size)
        self.a2 = act()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.d1 = nn.Dropout(p=drop)

        self.conv_layer3 = nn.Conv2d(in_channels=chan[1], out_channels=chan[2], kernel_size=kernel_size)
        self.a3 = act()
        self.conv_layer4 = nn.Conv2d(in_channels=chan[2], out_channels=chan[3], kernel_size=kernel_size)

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

        return out


class MergeModalitiesAtt2(nn.Module):
    def __init__(self, dim_inner, dim_face, dim_pose, dim_kq, act, value_transform=True):
        super(MergeModalitiesAtt2, self).__init__()

        self.dim_inner = dim_inner
        self.dim_face = dim_face
        self.dim_pose = dim_pose

        self.dimKQ = dim_kq

        # (the same as the input to the recurrent net)
        self.dimV = self.dim_face

        # define all necessary variables
        self.WQ = nn.Linear(self.dim_pose, self.dimKQ)
        self.WK = nn.Linear(self.dim_face, self.dimKQ)
        self.WV = nn.Linear(self.dim_face, self.dimV) if value_transform else nn.Identity()

        # self.b = nn.Parameter(torch.randn(self.dim_inner), requires_grad=True)
        self.feature_importance = nn.Identity()
        # self.act = act()

    def forward(self, face, pose):
        face_flat = face.view(face.shape[0], face.shape[1], face.shape[2]*face.shape[3])
        face_flat = face_flat.permute(0, 2, 1)

        q = self.WQ(pose)
        k = self.WK(face_flat)
        v = self.WV(face_flat)

        # multiplicative scoring function
        q = q.reshape(q.shape[0], 1, q.shape[1])
        dot_prod_sft = torch.matmul(q, k.transpose(-1, -2)).squeeze() / torch.sqrt(torch.tensor(self.dimKQ))
        # dot_prod_sft = torch.softmax(dot_prod, dim=0)
        out = self.feature_importance(dot_prod_sft)     # only due to fast extraction of features
        out = torch.matmul(v.transpose(-1, -2), out.unsqueeze(dim=-1)).squeeze()

        return out
