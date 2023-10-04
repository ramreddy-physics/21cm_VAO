import torch


class normalize(torch.nn.Module):
    def __init__(self):
        super(normalize, self).__init__()
        self.a = None
        self.b = None

    def forward(self, x):
        x_max = torch.max(x)
        x_min = torch.min(x)
        self.a = 2.0 / (x_max + x_min)
        self.b = -(x_max - x_min) / (x_max + x_min)
        return self.a * x + self.b


class encoder(torch.nn.Module):
    """
    Takes a 3D box with one channel as input, downsamples it into multiple channels using convolutions.
    The downsampled 3D box is then upsampled back to original shape by trans_conv.
    For example, here we have (N, 1, d, d, d) -> (N, 16, n1, n1, n1) -> (N, 1, d, d, d).
    For multiple conv, trans_conv layers, there seems to be an exploding gradients problem. Normalization Layers fix that problem.
    """

    def __init__(self, kernel_size, padding, input_shape):
        super(encoder, self).__init__()

        N, c, d, _, _ = input_shape

        self.conv1 = torch.nn.Conv3d(
            in_channels=c,
            out_channels=16,
            kernel_size=4,
            padding=2,
            padding_mode="zeros",
        )
        self.conv2 = torch.nn.Conv3d(
            in_channels=16,
            out_channels=16,
            kernel_size=4,
            padding=2,
            padding_mode="zeros",
        )

        self.trans_conv1 = torch.nn.ConvTranspose3d(
            in_channels=16,
            out_channels=c,
            kernel_size=4,
            padding=2,
            padding_mode="zeros",
        )
        self.trans_conv2 = torch.nn.ConvTranspose3d(
            in_channels=16,
            out_channels=16,
            kernel_size=4,
            padding=2,
            padding_mode="zeros",
        )

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.relu(x)

        x = self.trans_conv2(x)
        x = self.trans_conv1(x)

        return x


class planar_flow_layer(torch.nn.Module):
    """
    Input: 3D box x of the form (N, d, d, d)
    Output z = x + u*arctan(w*x + b)
    """

    def __init__(self, input_shape):
        super(planar_flow_layer, self).__init__()
        N, c, d, _, _ = input_shape
        self.u = torch.nn.Parameter(2 * torch.rand(size=(d, d, d)) - 1)
        self.w = torch.nn.Parameter(2 * torch.rand(size=(d, d, d)) - 1)
        self.b = torch.nn.Parameter(2 * torch.randn(1) - 1)
        self.normalizer = normalize()

    def forward(self, x: torch.tensor):
        y = torch.einsum("ijk, acijk-> ac", self.w, x)
        y = torch.tanh(y + self.b)
        y = torch.einsum("ijk, ac-> acijk", self.u, y)
        x = x + y
        return self.normalizer(x)


def create_planar_flow_model(input_shape, num_layers):
    """
    Return a Sequential Model of a stack of 'planar_flow_layer's
    """
    model = torch.nn.Sequential(
        *[planar_flow_layer(input_shape) for i in range(num_layers)]
    )
    return model


class planar_flow_merge_channels(torch.nn.Module):
    """
    Input: (N, c, d, d, d), c=1 corresponds to density field, c=2 corresponds to velocity
    Output: (N, 1, d, d, d), 3D box of 21cm field
    """

    def __init__(self, input_shape, num_layers_d=5, num_layers_v=5):
        super(planar_flow_merge_channels, self).__init__()
        self.NF_d = torch.nn.Sequential(
            *[planar_flow_layer(input_shape) for i in range(num_layers_d)]
        )
        self.NF_v = torch.nn.Sequential(
            *[planar_flow_layer(input_shape) for i in range(num_layers_v)]
        )

        self.w_d = torch.nn.Parameter(torch.randn(1))
        self.w_v = torch.nn.Parameter(torch.randn(1))
        self.b = torch.nn.Parameter(torch.randn(1))

    def forward(self, D: torch.tensor, V: torch.tensor):
        D = self.NF_d(D)
        V = self.NF_v(V)
        return torch.sigmoid(self.w_d * D + self.w_v * V + self.b)
