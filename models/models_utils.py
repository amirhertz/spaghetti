from custom_types import *
from abc import ABC
import math


def torch_no_grad(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            result = func(*args, **kwargs)
        return result

    return wrapper


class Model(nn.Module, ABC):

    def __init__(self):
        super(Model, self).__init__()
        self.save_model: Union[None, Callable[[nn.Module]]] = None

    def save(self, **kwargs):
        self.save_model(self, **kwargs)


class Concatenate(nn.Module):
    def __init__(self, dim):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)


class View(nn.Module):

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0, self.dim1 = dim0, dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class Dummy(nn.Module):

    def __init__(self, *args):
        super(Dummy, self).__init__()

    def forward(self, *args):
        return args[0]


class SineLayer(nn.Module):
    """
    From the siren repository
    https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb
    """
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.output_channels = out_features
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class MLP(nn.Module):

    def forward(self, x, *_):
        return self.net(x)

    def __init__(self, ch: Union[List[int], Tuple[int, ...]], act: nn.Module = nn.ReLU,
                 weight_norm=False):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(ch) - 1):
            layers.append(nn.Linear(ch[i], ch[i + 1]))
            if weight_norm:
                layers[-1] = nn.utils.weight_norm(layers[-1])
            if i < len(ch) - 2:
                layers.append(act(True))
        self.net = nn.Sequential(*layers)


class GMAttend(nn.Module):

    def __init__(self, hidden_dim: int):
        super(GMAttend, self).__init__()
        self.key_dim = hidden_dim // 8
        self.query_w = nn.Linear(hidden_dim, self.key_dim)
        self.key_w = nn.Linear(hidden_dim, self.key_dim)
        self.value_w = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = 1 / torch.sqrt(torch.tensor(self.key_dim, dtype=torch.float32))

    def forward(self, x):
        queries = self.query_w(x)
        keys = self.key_w(x)
        vals = self.value_w(x)
        attention = self.softmax(torch.einsum('bgqf,bgkf->bgqk', queries, keys))
        out = torch.einsum('bgvf,bgqv->bgqf', vals, attention)
        out = self.gamma * out + x
        return out


def recursive_to(item, device):
    if type(item) is T:
        return item.to(device)
    elif type(item) is tuple or type(item) is list:
        return [recursive_to(item[i], device) for i in range(len(item))]
    return item

