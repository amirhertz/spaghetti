from custom_types import *


def get_gm_support(gm, x):
    dim = x.shape[-1]
    mu, p, phi, eigen = gm
    sigma_det = eigen.prod(-1)
    eigen_inv = 1 / eigen
    sigma_inverse = torch.matmul(p.transpose(3, 4), p * eigen_inv[:, :, :, :, None]).squeeze(1)
    phi = torch.softmax(phi, dim=2)
    const_1 = phi / torch.sqrt((2 * np.pi) ** dim * sigma_det)
    distance = x[:, :, None, :] - mu
    mahalanobis_distance = - .5 * torch.einsum('bngd,bgdc,bngc->bng', distance, sigma_inverse, distance)
    const_2, _ = mahalanobis_distance.max(dim=2)  # for numeric stability
    mahalanobis_distance -= const_2[:, :, None]
    support = const_1 * torch.exp(mahalanobis_distance)
    return support, const_2


def gm_log_likelihood_loss(gms: TS, x: T, get_supports: bool = False,
                           mask: Optional[T] = None, reduction: str = "mean") -> Union[T, Tuple[T, TS]]:

    batch_size, num_points, dim = x.shape
    support, const = get_gm_support(gms, x)
    probs = torch.log(support.sum(dim=2)) + const
    if mask is not None:
        probs = probs.masked_select(mask=mask.flatten())
    if reduction == 'none':
        likelihood = probs.sum(-1)
        loss = - likelihood / num_points
    else:
        likelihood = probs.sum()
        loss = - likelihood / (probs.shape[0] * probs.shape[1])
    if get_supports:
        return loss, support
    return loss


def split_mesh_by_gmm(mesh: T_Mesh, gmm) -> Dict[int, T]:
    faces_split = {}
    vs, faces = mesh
    vs_mid_faces = vs[faces].mean(1)
    _, supports = gm_log_likelihood_loss(gmm, vs_mid_faces.unsqueeze(0), get_supports=True)
    supports = supports[0]
    label = supports.argmax(1)
    for i in range(gmm[1].shape[2]):
        select = label.eq(i)
        if select.any():
            faces_split[i] = faces[select]
        else:
            faces_split[i] = None
    return faces_split


def flatten_gmm(gmm: TS) -> T:
    b, gp, g, _ = gmm[0].shape
    mu, p, phi, eigen = [item.view(b, gp * g, *item.shape[3:]) for item in gmm]
    p = p.reshape(*p.shape[:2], -1)
    z_gmm = torch.cat((mu, p, phi.unsqueeze(-1), eigen), dim=2)
    return z_gmm
