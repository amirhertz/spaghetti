import constants
import functools
from scipy.spatial.transform.rotation import Rotation
from custom_types import *


def quat_to_rot(q):
    shape = q.shape
    q = q.view(-1, 4)
    q_sq = 2 * q[:, :, None] * q[:, None, :]
    m00 = 1 - q_sq[:, 1, 1] - q_sq[:, 2, 2]
    m01 = q_sq[:, 0, 1] - q_sq[:, 2, 3]
    m02 = q_sq[:, 0, 2] + q_sq[:, 1, 3]

    m10 = q_sq[:, 0, 1] + q_sq[:, 2, 3]
    m11 = 1 - q_sq[:, 0, 0] - q_sq[:, 2, 2]
    m12 = q_sq[:, 1, 2] - q_sq[:, 0, 3]

    m20 = q_sq[:, 0, 2] - q_sq[:, 1, 3]
    m21 = q_sq[:, 1, 2] + q_sq[:, 0, 3]
    m22 = 1 - q_sq[:, 0, 0] - q_sq[:, 1, 1]
    r = torch.stack((m00, m01, m02, m10, m11, m12, m20, m21, m22), dim=1)
    r = r.view(*shape[:-1], 3, 3)
    return r


def rot_to_quat(r):
    shape = r.shape
    r = r.view(-1, 3, 3)
    qw = .5 * (1 + r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]).sqrt()
    qx = (r[:, 2, 1] - r[:, 1, 2]) / (4 * qw)
    qy = (r[:, 0, 2] - r[:, 2, 0]) / (4 * qw)
    qz = (r[:, 1, 0] - r[:, 0, 1]) / (4 * qw)
    q = torch.stack((qx, qy, qz, qw), -1)
    q = q.view(*shape[:-2], 4)
    return q


@functools.lru_cache(10)
def get_rotation_matrix(theta: float, axis: float, degree: bool = False) -> ARRAY:
    if degree:
        theta = theta * np.pi / 180
    rotate_mat = np.eye(3)
    rotate_mat[axis, axis] = 1
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotate_mat[(axis + 1) % 3, (axis + 1) % 3] = cos_theta
    rotate_mat[(axis + 2) % 3, (axis + 2) % 3] = cos_theta
    rotate_mat[(axis + 1) % 3, (axis + 2) % 3] = sin_theta
    rotate_mat[(axis + 2) % 3, (axis + 1) % 3] = -sin_theta
    return rotate_mat


def get_random_rotation(batch_size: int) -> T:
    r = Rotation.random(batch_size).as_matrix().astype(np.float32)
    Rotation.random()
    return torch.from_numpy(r)


def rand_bounded_rotation_matrix(cache_size: int, theta_range: float = .1):

    def create_cache():
        # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
        with torch.no_grad():
            theta, phi, z = torch.rand(cache_size, 3).split((1, 1, 1), dim=1)
            theta = (2 * theta - 1) * theta_range + 1
            theta = np.pi * theta   # Rotation about the pole (Z).
            phi = phi * 2 * np.pi  # For direction of pole deflection.
            z = 2 * z * theta_range # For magnitude of pole deflection.
            r = z.sqrt()
            v = torch.cat((torch.sin(phi) * r, torch.cos(phi) * r, torch.sqrt(2.0 - z)), dim=1)
            st = torch.sin(theta).squeeze(1)
            ct = torch.cos(theta).squeeze(1)
            rot_ = torch.zeros(cache_size, 3, 3)
            rot_[:, 0, 0] = ct
            rot_[:, 1, 1] = ct
            rot_[:, 0, 1] = st
            rot_[:, 1, 0] = -st
            rot_[:, 2, 2] = 1
            rot = (torch.einsum('ba,bd->bad', v, v) - torch.eye(3)[None, :, :]).bmm(rot_)
            det = rot.det()
        assert (det.gt(0.99) * det.lt(1.0001)).all().item()
        return rot

    def get_batch_rot(batch_size):
        nonlocal cache
        select = torch.randint(cache_size, size=(batch_size,))
        return cache[select]

    cache = create_cache()

    return get_batch_rot


def transform_rotation(points: T, one_axis=False, max_angle=-1):
    r = get_random_rotation(one_axis, max_angle)
    transformed = torch.einsum('nd,rd->nr', points, r)
    return transformed


def tb_to_rot(abc: T) -> T:
    c, s = torch.cos(abc), torch.sin(abc)
    aa = c[:, 0] * c[:, 1]
    ab = c[:, 0] * s[:, 1] * s[:, 2] - c[:, 2] * s[:, 0]
    ac = s[:, 0] * s[:, 2] + c[:, 0] * c[:, 2] * s[:, 1]

    ba = c[:, 1] * s[:, 0]
    bb = c[:, 0] * c[:, 2] + s.prod(-1)
    bc = c[:, 2] * s[:, 0] * s[:, 1] - c[:, 0] * s[:, 2]

    ca = -s[:, 1]
    cb = c[:, 1] * s[:, 2]
    cc = c[:, 1] * c[:, 2]
    return torch.stack((aa, ab, ac, ba, bb, bc, ca, cb, cc), 1).view(-1, 3, 3)


def rot_to_tb(rot: T) -> T:
    sy = torch.sqrt(rot[:, 0, 0] * rot[:, 0, 0] + rot[:, 1, 0] * rot[:, 1, 0])
    out = torch.zeros(rot.shape[0], 3, device = rot.device)
    mask = sy.gt(1e-6)
    z = torch.atan2(rot[mask, 2, 1], rot[mask, 2, 2])
    y = torch.atan2(-rot[mask, 2, 0], sy[mask])
    x = torch.atan2(rot[mask, 1, 0], rot[mask, 0, 0])
    out[mask] = torch.stack((x, y, z), dim=1)
    if not mask.all():
        mask = ~mask
        z = torch.atan2(-rot[mask, 1, 2], rot[mask, 1, 1])
        y = torch.atan2(-rot[mask, 2, 0], sy[mask])
        x = torch.zeros(x.shape)
        out[mask] = torch.stack((x, y, z), dim=1)
    return out


def apply_gmm_affine(gmms: TS, affine: T):
    mu, p, phi, eigen = gmms
    if affine.dim() == 2:
        affine = affine.unsqueeze(0).expand(mu.shape[0], *affine.shape)
    mu_r = torch.einsum('bad, bpnd->bpna', affine, mu)
    p_r = torch.einsum('bad, bpncd->bpnca', affine, p)
    return mu_r, p_r, phi, eigen


def get_reflection(reflect_axes: Tuple[bool, ...]) -> T:
    reflect = torch.eye(constants.DIM)
    for i in range(constants.DIM):
        if reflect_axes[i]:
            reflect[i, i] = -1
    return reflect


def get_tait_bryan_from_p(p: T) -> T:
    # p = p.squeeze(1)
    shape = p.shape
    rot = p.reshape(-1, 3, 3).permute(0, 2, 1)
    angles = rot_to_tb(rot)
    angles = angles / np.pi
    angles[:, 1] = angles[:, 1] * 2
    angles = angles.view(*shape[:2], 3)
    return angles
