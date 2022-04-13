from options import Options
from models import models_utils, transformer
import constants
from custom_types import *
from torch import distributions
import math
from utils import files_utils

def dot(x, y, dim=3):
    return torch.sum(x * y, dim=dim)


def remove_projection(v_1, v_2):
    proj = (dot(v_1, v_2) / dot(v_2, v_2))
    return v_1 - proj[:, :, :, None] * v_2


def get_p_direct(splitted: TS) -> T:
    raw_base = []
    for i in range(constants.DIM):
        u = splitted[i]
        for j in range(i):
            u = remove_projection(u, raw_base[j])
        raw_base.append(u)
    p = torch.stack(raw_base, dim=3)
    p = p / torch.norm(p, p=2, dim=4)[:, :, :, :, None]  # + self.noise[None, None, :, :]
    return p


def split_gm(splitted: TS) -> TS:
    p = get_p_direct(splitted)
    # eigenvalues
    eigen = splitted[-3] ** 2 + constants.EPSILON
    mu = splitted[-2]
    phi = splitted[-1].squeeze(3)
    return mu, p, phi, eigen


class DecompositionNetwork(nn.Module):

    def forward_bottom(self, x):
        return self.l1(x).view(-1, self.bottom_width, self.embed_dim)

    def forward_upper(self, x):
        return self.to_zb(x)

    def forward(self, x):
        x = self.forward_bottom(x)
        x = self.forward_upper(x)
        return x

    def __init__(self, opt: Options, act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm):
        super(DecompositionNetwork, self).__init__()
        self.bottom_width = opt.num_gaussians
        self.embed_dim = opt.dim_h
        self.l1 = nn.Linear(opt.dim_z, self.bottom_width * opt.dim_h)
        if opt.decomposition_network == 'mlp':

            self.to_zb = models_utils.MLP((opt.dim_h, *([2 * opt.dim_h] * opt.decomposition_num_layers), opt.dim_h))
        else:
            self.to_zb = transformer.Transformer(opt.dim_h, opt.num_heads, opt.num_layers, act=act,
                                                       norm_layer=norm_layer)


class OccupancyMlP(nn.Module):
    ## base on DeepSDF https://github.com/facebookresearch/DeepSDF
    def forward(self, x, z):
        x_ = x = torch.cat((x, z), dim=-1)
        for i, layer in enumerate(self.layers):
            if layer == self.latent_in:
                x = torch.cat([x, x_], 2)
            x = layer(x)
            if i < len(self.layers) - 2:
                x = self.relu(x)
                # x = self.dropout(self.relu(x))
            # files_utils.save_pickle(x.detach().cpu(), f"/home/amirh/projects/spaghetti_private/assets/debug/out_{i}")
        return x

    def __init__(self, opt: Options):
        super(OccupancyMlP, self).__init__()
        dim_in = 2 * (opt.pos_dim + constants.DIM)
        dims = [dim_in] + opt.head_occ_size * [dim_in] + [1]
        self.latent_in = opt.head_occ_size // 2 + opt.head_occ_size % 2
        dims[self.latent_in] += dims[0]
        self.dropout = nn.Dropout(.2)
        self.relu = nn.ReLU(True)
        layers = []
        for i in range(0, len(dims) - 1):
            layers.append(nn.utils.weight_norm(nn.Linear(dims[i], dims[i + 1])))
        self.layers = nn.ModuleList(layers)


class OccupancyNetwork(nn.Module):

    def get_pos(self, coords: T):
        pos = self.pos_encoder(coords)
        pos = torch.cat((coords, pos), dim=2)
        return pos

    def forward_attention(self, coords: T, zh: T, mask: Optional[T] = None, alpha: TN = None) -> TS:
        pos = self.get_pos(coords)
        _, attn = self.occ_transformer.forward_with_attention(pos, zh, mask, alpha)
        return attn

    def forward(self, coords: T, zh: T,  mask: TN = None, alpha: TN = None) -> T:
        pos = self.get_pos(coords)
        x = self.occ_transformer(pos, zh, mask, alpha)
        out = self.occ_mlp(pos, x)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out

    def __init__(self, opt: Options):
        super(OccupancyNetwork, self).__init__()
        self.pos_encoder = models_utils.SineLayer(constants.DIM, opt.pos_dim, is_first=True)

        if hasattr(opt, 'head_occ_type') and opt.head_occ_type == 'skip':
            self.occ_mlp = OccupancyMlP(opt)
        else:
            self.occ_mlp = models_utils.MLP([(opt.pos_dim + constants.DIM)] +
                                            [opt.dim_h] * opt.head_occ_size + [1])
        self.occ_transformer = transformer.Transformer(opt.pos_dim + constants.DIM,
                                                       opt.num_heads_head, opt.num_layers_head,
                                                       dim_ref=opt.dim_h)

class DecompositionControl(models_utils.Model):

    def forward_bottom(self, x):
        z_bottom = self.decomposition.forward_bottom(x)
        return z_bottom

    def forward_upper(self, x):
        x = self.decomposition.forward_upper(x)
        return x

    def forward_split(self, x: T) -> Tuple[T, TS]:
        b = x.shape[0]
        raw_gmm = self.to_gmm(x).unsqueeze(1)
        gmms = split_gm(torch.split(raw_gmm, self.split_shape, dim=3))
        zh = self.to_s(x)
        zh = zh.view(b, -1, zh.shape[-1])
        return zh, gmms

    @staticmethod
    def apply_gmm_affine(gmms: TS, affine: T):
        mu, p, phi, eigen = gmms
        if affine.dim() == 2:
            affine = affine.unsqueeze(0).expand(mu.shape[0], *affine.shape)
        mu_r = torch.einsum('bad, bpnd->bpna', affine, mu)
        p_r = torch.einsum('bad, bpncd->bpnca', affine, p)
        return mu_r, p_r, phi, eigen

    @staticmethod
    def concat_gmm(gmm_a: TS, gmm_b: TS):
        out = []
        num_gaussians = gmm_a[0].shape[2] // 2
        for element_a, element_b in zip(gmm_a, gmm_b):
            out.append(torch.cat((element_a[:, :, :num_gaussians], element_b[:, :, :num_gaussians]), dim=2))
        return out

    def forward_mid(self, zs) -> Tuple[T, TS]:
        zh, gmms = self.forward_split(zs)
        if self.reflect is not None:
            gmms_r = self.apply_gmm_affine(gmms, self.reflect)
            gmms = self.concat_gmm(gmms, gmms_r)
        return zh, gmms

    def forward_low(self, z_init):
        zs = self.decomposition(z_init)
        return zs

    def forward(self, z_init) -> Tuple[T, TS]:
        zs = self.forward_low(z_init)
        zh, gmms = self.forward_mid(zs)
        return zh, gmms

    @staticmethod
    def get_reflection(reflect_axes: Tuple[bool, ...]):
        reflect = torch.eye(constants.DIM)
        for i in range(constants.DIM):
            if reflect_axes[i]:
                reflect[i, i] = -1
        return reflect

    def __init__(self, opt: Options):
        super(DecompositionControl, self).__init__()
        if sum(opt.symmetric) > 0:
            reflect = self.get_reflection(opt.symmetric)
            self.register_buffer("reflect", reflect)
        else:
            self.reflect = None
        self.split_shape = tuple((constants.DIM + 2) * [constants.DIM] + [1])
        self.decomposition = DecompositionNetwork(opt)
        self.to_gmm = nn.Linear(opt.dim_h, sum(self.split_shape))
        self.to_s = nn.Linear(opt.dim_h, opt.dim_h)


class Spaghetti(models_utils.Model):

    def get_z(self, item: T):
        return self.z(item)

    @staticmethod
    def interpolate_(z, num_between: Optional[int] = None):
        if num_between is None:
            num_between = z.shape[0]
        alphas = torch.linspace(0, 1, num_between, device=z.device)
        while alphas.dim() != z.dim():
            alphas.unsqueeze_(-1)
        z_between = alphas * z[1:2] + (- alphas + 1) * z[:1]
        return z_between

    def interpolate_higher(self, z: T, num_between: Optional[int] = None):
        z_between = self.interpolate_(z, num_between)
        zh, gmms = self.decomposition_control.forward_split(self.decomposition_control.forward_upper(z_between))
        return zh, gmms

    def interpolate(self, item_a: int, item_b: int, num_between: int):
        items = torch.tensor((item_a, item_b), dtype=torch.int64, device=self.device)
        z = self.get_z(items)
        z_between = self.interpolate_(z, num_between)
        zh, gmms = self.decomposition_control(z_between)
        return zh, gmms

    def get_disentanglement(self, items: T):
        z_a = self.get_z(items)
        z_b = self.decomposition_control.forward_bottom(z_a)
        zh, gmms = self.decomposition_control.forward_split(self.decomposition_control.forward_upper(z_b))
        return z_a, z_b, zh, gmms

    def get_embeddings(self, item: T):
        z = self.get_z(item)
        zh, gmms = self.decomposition_control(z)
        return zh, z, gmms

    def merge_zh_step_a(self, zh, gmms):
        b, gp, g, _ = gmms[0].shape
        mu, p, phi, eigen = [item.view(b, gp * g, *item.shape[3:]) for item in gmms]
        p = p.reshape(*p.shape[:2], -1)
        z_gmm = torch.cat((mu, p, phi.unsqueeze(-1), eigen), dim=2).detach()
        z_gmm = self.from_gmm(z_gmm)
        zh_ = zh + z_gmm
        return zh_

    def merge_zh(self, zh, gmms, mask: Optional[T] = None) -> TNS:
        zh_ = self.merge_zh_step_a(zh, gmms)
        zh_, attn = self.mixing_network.forward_with_attention(zh_, mask=mask)
        return zh_, attn

    def forward_b(self, x, zh, gmms, mask: Optional[T] = None) -> T:
        zh, _ = self.merge_zh(zh, gmms, mask)
        return self.occupancy_network(x, zh, mask)

    def forward_a(self, item: T):
        zh, z, gmms = self.get_embeddings(item)
        return zh, z, gmms

    def get_attention(self, x, item) -> TS:
        zh, z, gmms = self.forward_a(item)
        zh, _ = self.merge_zh(zh, gmms)
        return self.occupancy_network.forward_attention(x, zh)

    def forward(self, x, item: T) -> Tuple[T, T, TS, T]:
        zh, z, gmms = self.forward_a(item)
        return self.forward_b(x, zh, gmms), z, gmms, zh

    def forward_mid(self, x: T, zh: T) -> Tuple[T, TS]:
        zh, gmms = self.decomposition_control.forward_mid(zh)
        return self.forward_b(x, zh, gmms), gmms

    def get_random_embeddings(self, num_items: int):
        if self.dist is None:
            weights = self.z.weight.clone().detach()
            mean = weights.mean(0)
            weights = weights - mean[None, :]
            cov = torch.einsum('nd,nc->dc', weights, weights) / (weights.shape[0] - 1)
            self.dist = distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix=cov)
        z_init = self.dist.sample((num_items,))
        return z_init

    def random_samples(self, num_items: int):
        z_init = self.get_random_embeddings(num_items)
        zh, gmms = self.decomposition_control(z_init)
        return zh, gmms

    def __init__(self, opt: Options):
        super(Spaghetti, self).__init__()
        self.device = opt.device
        self.opt = opt
        self.z = nn.Embedding(opt.dataset_size, opt.dim_z)
        torch.nn.init.normal_(
            self.z.weight.data,
            0.0,
            1. / math.sqrt(opt.dim_z),
        )
        self.decomposition_control = DecompositionControl(opt)
        self.occupancy_network = OccupancyNetwork(opt)
        self.from_gmm = nn.Linear(sum(self.decomposition_control.split_shape), opt.dim_h)
        if opt.use_encoder:
            self.mixing_network = transformer.Transformer(opt.dim_h, opt.num_heads, opt.num_layers,
                                                              act=nnf.relu, norm_layer=nn.LayerNorm)
        else:
            self.mixing_network = transformer.DummyTransformer()
        self.dist = None
