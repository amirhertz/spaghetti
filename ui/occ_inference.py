from custom_types import *
from options import Options
from utils import train_utils, mcubes_meshing, files_utils, mesh_utils
from models.occ_gmm import Spaghetti
from models import models_utils


class Inference:

    def get_occ_fun(self, z: T):

        def forward(x: T) -> T:
            nonlocal z
            x = x.unsqueeze(0)
            out = self.model.occupancy_network(x, z)[0, :]
            out = 2 * out.sigmoid_() - 1
            return out

        if z.dim() == 2:
            z = z.unsqueeze(0)
        return forward

    @models_utils.torch_no_grad
    def get_mesh(self, z: T, res: int) -> Optional[T_Mesh]:
        mesh = self.meshing.occ_meshing(self.get_occ_fun(z), res=res)
        return mesh

    def plot_occ(self, z: Union[T, TS], z_base, gmms: Optional[TS], fixed_items: T,
                 folder_name: str, res=200, verbose=False):
        for i in range(len(z)):
            mesh = self.get_mesh(z[i], res)
            name = f'{fixed_items[i]:04d}'
            if mesh is not None:
                files_utils.export_mesh(mesh, f'{self.opt.cp_folder}/{folder_name}/occ/{name}')
                files_utils.save_pickle(z_base[i].detach().cpu(), f'{self.opt.cp_folder}/{folder_name}/occ/{name}')
                if gmms is not None:
                    files_utils.export_gmm(gmms, i, f'{self.opt.cp_folder}/{folder_name}/occ/{name}')
            if verbose:
                print(f'done {i + 1:d}/{len(z):d}')

    def load_file(self, info_path, disclude: Optional[List[int]] = None):
        info = files_utils.load_pickle(''.join(info_path))
        keys = list(info['ids'].keys())
        items = map(lambda x: int(x.split('_')[1]) if type(x) is str else x, keys)
        items = torch.tensor(list(items), dtype=torch.int64, device=self.device)
        zh, _, gmms_sanity, _ = self.model.get_embeddings(items)
        gmms = [item for item in info['gmm']]
        zh_ = []
        split = []
        gmm_mask = torch.ones(gmms[0].shape[2], dtype=torch.bool)
        counter = 0
        # gmms_ = [[] for _ in range(len(gmms))]
        for i, key in enumerate(keys):
            gaussian_inds = info['ids'][key]
            if disclude is not None:
                for j in range(len(gaussian_inds)):
                    gmm_mask[j + counter] = gaussian_inds[j] not in disclude
                counter += len(gaussian_inds)
                gaussian_inds = [ind for ind in gaussian_inds if ind not in disclude]
                info['ids'][key] = gaussian_inds
            gaussian_inds = torch.tensor(gaussian_inds, dtype=torch.int64)
            zh_.append(zh[i, gaussian_inds])
            split.append(len(split) + torch.ones(len(info['ids'][key]), dtype=torch.int64, device=self.device))
        zh_ = torch.cat(zh_, dim=0).unsqueeze(0).to(self.device)
        gmms = [item[:, :, gmm_mask].to(self.device) for item in info['gmm']]
        return zh_, gmms, split, info['ids']

    @models_utils.torch_no_grad
    def get_z_from_file(self, info_path):
        zh_, gmms, split, _ = self.load_file(info_path)
        zh_ = self.model.merge_zh_step_a(zh_, [gmms])
        zh, _ = self.model.affine_transformer.forward_with_attention(zh_)
        # gmms_ = [torch.cat(item, dim=1).unsqueeze(0) for item in gmms_]
        # zh, _ = self.model.merge_zh(zh_, [gmms])
        return zh, zh_, gmms, torch.cat(split)

    def plot_from_info(self, info_path, res):
        zh, zh_, gmms, split = self.get_z_from_file(info_path)
        mesh = self.get_mesh(zh[0], res, gmms)
        if mesh is not None:
            attention = self.get_attention_faces(mesh, zh, fixed_z=split)
        else:
            attention = None
        return mesh, attention

    @staticmethod
    def combine_and_pad(zh_a: T, zh_b: T) -> Tuple[T, TN]:
        if zh_a.shape[1] == zh_b.shape[1]:
            mask = None
        else:
            pad_length = max(zh_a.shape[1], zh_b.shape[1])
            mask = torch.zeros(2, pad_length, device=zh_a.device, dtype=torch.bool)
            padding = torch.zeros(1, abs(zh_a.shape[1] - zh_b.shape[1]), zh_a.shape[-1], device=zh_a.device)
            if zh_a.shape[1] > zh_b.shape[1]:
                mask[1, zh_b.shape[1]:] = True
                zh_b = torch.cat((zh_b, padding), dim=1)
            else:
                mask[0, zh_a.shape[1]:] = True
                zh_a = torch.cat((zh_a, padding), dim=1)
        return torch.cat((zh_a, zh_b), dim=0), mask

    @staticmethod
    def get_intersection_z(z_a: T, z_b: T) -> T:
        diff = (z_a[0, :, None, :] - z_b[0, None]).abs().sum(-1)
        diff_a = diff.min(1)[0].lt(.1)
        diff_b = diff.min(0)[0].lt(.1)
        if diff_a.shape[0] != diff_b.shape[0]:
            padding = torch.zeros(abs(diff_a.shape[0] - diff_b.shape[0]), device=z_a.device, dtype=torch.bool)
            if diff_a.shape[0] > diff_b.shape[0]:
                diff_b = torch.cat((diff_b, padding))
            else:
                diff_a = torch.cat((diff_a, padding))
        return torch.cat((diff_a, diff_b))

    def get_attention_points(self, vs: T, zh: T, mask: TN = None, alpha: TN = None):
        vs = vs.unsqueeze(0)
        attention = self.model.occupancy_network.forward_attention(vs, zh, mask=mask, alpha=alpha)
        attention = torch.stack(attention, 0).mean(0).mean(-1)
        attention = attention.permute(1, 0, 2).reshape(attention.shape[1], -1)
        attention_max = attention.argmax(-1)
        return attention_max

    @models_utils.torch_no_grad
    def get_attention_faces(self, mesh: T_Mesh, zh: T, mask: TN = None, fixed_z: TN = None, alpha: TN = None):
        coords = mesh[0][mesh[1]].mean(1).to(zh.device)

        attention_max = self.get_attention_points(coords, zh, mask, alpha)
        if fixed_z is not None:
            attention_select = fixed_z[attention_max].cpu()
        else:
            attention_select = attention_max
        return attention_select

    @models_utils.torch_no_grad
    def plot_folder(self, *folders, res: int = 256):
        logger = train_utils.Logger()
        for folder in folders:
            paths = files_utils.collect(folder, '.pkl')
            logger.start(len(paths))
            for path in paths:
                name = path[1]
                out_path = f"{self.opt.cp_folder}/from_ui/{name}"
                mesh, colors = self.plot_from_info(path, res)
                if mesh is not None:
                    files_utils.export_mesh(mesh, out_path)
                    files_utils.export_list(colors.tolist(), f"{out_path}_faces")
                logger.reset_iter()
            logger.stop()

    def get_zh_from_idx(self, items: T):
        zh, _, gmms, __ = self.model.get_embeddings(items.to(self.device))
        zh, attn_b = self.model.merge_zh(zh, gmms)
        return zh, gmms

    @property
    def device(self):
        return self.opt.device

    def get_new_ids(self, folder_name, nums_sample):
        names = [int(path[1]) for path in files_utils.collect(f'{self.opt.cp_folder}/{folder_name}/occ/', '.obj')]
        ids = torch.arange(nums_sample)
        if len(names) == 0:
            return ids + self.opt.dataset_size
        return ids + max(max(names) + 1, self.opt.dataset_size)

    @models_utils.torch_no_grad
    def random_plot(self, folder_name: str, nums_sample, res=200, verbose=False):
        zh_base, gmms = self.model.random_samples(nums_sample)
        zh, attn_b = self.model.merge_zh(zh_base, gmms)
        numbers = self.get_new_ids(folder_name, nums_sample)
        self.plot_occ(zh, zh_base, gmms, numbers, folder_name, verbose=verbose, res=res)

    @models_utils.torch_no_grad
    def plot(self, folder_name: str, nums_sample: int, verbose=False, res: int = 200):
        if self.model.opt.dataset_size < nums_sample:
            fixed_items = torch.arange(self.model.opt.dataset_size)
        else:
            fixed_items = torch.randint(low=0, high=self.opt.dataset_size, size=(nums_sample,))
        zh_base, _, gmms = self.model.get_embeddings(fixed_items.to(self.device))
        zh, attn_b = self.model.merge_zh(zh_base, gmms)
        self.plot_occ(zh, zh_base, gmms, fixed_items, folder_name, verbose=verbose, res=res)

    def get_mesh_from_mid(self, gmm, included: T, res: int) -> Optional[T_Mesh]:
        if self.mid is None:
            return None
        gmm = [elem.to(self.device) for elem in gmm]
        included = included.to(device=self.device)
        mid_ = self.mid[included[:, 0], included[:, 1]].unsqueeze(0)
        zh = self.model.merge_zh(mid_, gmm)[0]
        mesh = self.get_mesh(zh[0], res)
        return mesh

    def set_items(self, items: T):
        self.mid = items.to(self.device)

    def __init__(self, opt: Options):
        self.opt = opt
        model: Tuple[Spaghetti, Options] = train_utils.model_lc(opt)
        self.model, self.opt = model
        self.model.eval()
        self.mid: Optional[T] = None
        self.gmms: Optional[TN] = None
        self.meshing = mcubes_meshing.MarchingCubesMeshing(self.device, scale=1.)

