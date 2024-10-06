import torch.nn as nn
import torch


class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=None):
        super().__init__()
        padding = dilation
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)

        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()

        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0, )

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None):
        super().__init__()

        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm)
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim=263,
        output_dim=16,
        emb_dim_encoder=(256, 192, 128, 64, 32, 16),
        input_fps=20,
        rvq_fps=50,
        dilation_growth_rate=2,
        depth_per_res_block=6,
        activation='relu',
        norm=None,
        **kwargs
    ):
        super().__init__()

        self.input_fps = input_fps
        self.rvq_fps = rvq_fps

        self.init_conv = nn.Sequential(
            nn.Conv1d(input_dim, emb_dim_encoder[0], 3, 1, 1),
            nn.ReLU()
        )

        blocks = []

        for i in range(len(emb_dim_encoder) - 1):
            in_channel = emb_dim_encoder[i]
            out_channel = emb_dim_encoder[i + 1]

            block = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, 3, 1, 1),
                Resnet1D(out_channel, n_depth=depth_per_res_block, dilation_growth_rate=dilation_growth_rate,
                         activation=activation, norm=norm)
            )
            blocks.append(block)

        self.resnet_model = nn.Sequential(*blocks)

        self.post_conv = nn.Conv1d(emb_dim_encoder[-1], output_dim, 3, 1, 1)

    def forward(self, x):
        input_T = x.shape[-1]
        target_T = round(input_T / self.input_fps * self.rvq_fps)
        x = nn.functional.interpolate(x, size=target_T, mode='linear')

        x = self.init_conv(x)
        x = self.resnet_model(x)
        x = self.post_conv(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim=263,
        output_dim=16,
        emb_dim_decoder=(16, 32, 64, 128, 192, 256),
        input_fps=20,
        rvq_fps=50,
        dilation_growth_rate=2,
        depth_per_res_block=6,
        activation='relu',
        norm=None,
        **kwargs
    ):
        super().__init__()

        self.input_fps = input_fps
        self.rvq_fps = rvq_fps

        self.init_conv = nn.Sequential(
            nn.Conv1d(output_dim, emb_dim_decoder[0], 3, 1, 1),
            nn.ReLU()
        )

        blocks = []
        for i in range(len(emb_dim_decoder) - 1):
            in_channel = emb_dim_decoder[i]
            out_channel = emb_dim_decoder[i + 1]

            block = nn.Sequential(
                Resnet1D(in_channel, depth_per_res_block,
                         dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Conv1d(in_channel, out_channel, 3, 1, 1)
            )
            blocks.append(block)
        self.resnet_block = nn.Sequential(*blocks)

        self.post_conv = nn.Sequential(
            nn.Conv1d(emb_dim_decoder[-1], emb_dim_decoder[-1], 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(emb_dim_decoder[-1], input_dim, 3, 1, 1)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.resnet_block(x)
        x = self.post_conv(x)

        current_T = x.shape[-1]
        target_T = round(current_T / self.rvq_fps * self.input_fps)
        x = nn.functional.interpolate(x, target_T, mode='linear')

        return x
