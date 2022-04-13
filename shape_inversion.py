import abc
from custom_types import *
import options
import constants
from utils import train_utils, files_utils, mcubes_meshing, mesh_utils, myparse
from models import models_utils, occ_loss, gm_utils
from ui import occ_inference


class MeshSampler(Dataset):

    def __len__(self):
        return self.labels.shape[1]

    def __getitem__(self, item):
        return self.points[:, item], self.labels[:, item]

    def get_batch(self, item, batch_size):
        return self.points[:, item], self.labels[:, item]

    def get_surface_points(self, num_samples: int):
        counter = 0
        points = []
        while counter < num_samples:
            points_ = mesh_utils.sample_on_mesh(self.mesh, num_samples)[0]
            points.append(points_)
            counter += points_.shape[0]
        points = torch.cat(points, dim=0)
        if counter > num_samples:
            points = points[:num_samples]
        return points

    def get_labels(self, points: T) -> T:
        return torch.from_numpy(mesh_utils.get_fast_inside_outside(self.mesh, points.numpy()))

    def init_in_out(self, total=6e5, max_trials=50) -> TS:
        split_size = int(total // 4)
        on_surface_points = self.get_surface_points(split_size)
        near_points_a = on_surface_points + torch.randn(on_surface_points.shape) * .007
        near_points_b = on_surface_points + torch.randn(on_surface_points.shape) * .02
        inside_points_ = (torch.rand(split_size, 3) * 2 - 1) * self.global_scale
        random_points = inside_points_
        labels_near_a = self.get_labels(near_points_a)
        labels_near_b = self.get_labels(near_points_b)
        labels_inside = labels_random = self.get_labels(random_points)
        inside_points = [near_points_b[~labels_near_b]]
        counter_inside = inside_points[-1].shape[0]
        trial = 0
        while counter_inside < split_size and trial < max_trials:
            if trial == max_trials - 1:
                self.error = True
                return torch.zeros(1), torch.zeros(1)
            inside_points.append(inside_points_[~labels_inside])
            counter_inside += inside_points[-1].shape[0]
            if counter_inside < split_size:
                inside_points_ = self.mesh_bb[0][None, :] + self.mesh_bb[1][None, :] * torch.rand(split_size, 3)
                labels_inside = self.get_labels(inside_points_)
            trial += 1
        inside_points = torch.cat(inside_points, dim=0)[:split_size]
        inside_points = inside_points[torch.rand(inside_points.shape[0]).argsort()]
        all_points = torch.stack((near_points_a, near_points_b, random_points, inside_points), dim=0)
        labels = torch.stack((labels_near_a, labels_near_b, labels_random), dim=0)
        return all_points, labels

    def reset(self):
        self.points, self.labels = self.init_samples()

    def __init__(self, path, inside_outside=True, num_samples=6e5):
        mesh = files_utils.load_mesh(path)
        self.global_scale = 1
        mesh = mesh_utils.to_unit_sphere(mesh, scale=.90)
        mesh_bb = mesh[0].min(0)[0], mesh[0].max(0)[0]
        self.mesh_bb = mesh_bb[0], mesh_bb[1] - mesh_bb[0]
        self.mesh = mesh
        self.split = (2, 2, 2)
        self.points, self.labels = self.init_in_out(total=num_samples)


def get_data_loader(mesh_path: str) -> DataLoader:
    ds = MeshSampler(mesh_path)
    dl = DataLoader(ds, batch_size=5000, num_workers=4 if constants.DEBUG else 0,
                    shuffle=not constants.DEBUG, drop_last=False)
    return dl


class ProjectionType(Enum):
    LowProjection = "low_projection"
    HighProjection = "high_projection"
    E2EProjection = "e2e_projection"


class MeshProjection(occ_inference.Inference, abc.ABC):

    def train_iter(self, labels, samples_occ, samples_gmm):
        raise NotImplementedError

    def train_epoch(self, epoch: int):
        self.logger.start(len(self.dl), tag=f'{self.opt.tag}_projection')
        for data in self.dl:
            samples, labels = map(lambda x: x.to(self.device), data)
            samples_occ, samples_gmm = samples[:, :3].reshape(1, -1, 3), samples[:, 3].unsqueeze(0)
            labels = labels.unsqueeze(0)
            loss = self.train_iter(labels, samples_occ, samples_gmm)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.logger.reset_iter()
        return self.logger.stop()

    @property
    def device(self):
        return self.opt.device

    def init_embeddings(self):
        embeddings = self.model.get_random_embeddings(1)
        embeddings_wrap = nn.Embedding(1, embeddings.shape[1]).to(self.device)
        embeddings_wrap.weight.data = embeddings.data
        optimizer = Optimizer(embeddings_wrap.parameters(), lr=1e-7)
        return embeddings_wrap, optimizer

    def __init__(self, opt: options.Options, mesh_path: str, folder_out):
        super(MeshProjection, self).__init__(opt)
        self.opt = opt.load()
        self.logger = train_utils.Logger()
        self.dl: DataLoader = get_data_loader(mesh_path)
        self.embeddings, self.optimizer = self.init_embeddings()
        self.warm_up_scheduler = train_utils.LinearWarmupScheduler(self.optimizer, 1e-3, 100)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, .5)
        self.last_loss = 1000000
        self.meshing = mcubes_meshing.MarchingCubesMeshing(self.device, scale=1, min_res=200)
        self.criterion = occ_loss.occupancy_bce
        self.gmm_criterion = gm_utils.gm_log_likelihood_loss
        self.folder_name = folder_out


class MeshProjectionMid(MeshProjection):

    def switch(self, item_a, item_b):
        return item_a if self.projection_type is ProjectionType.LowProjection else item_b
    
    @property
    def loss_key(self):
        return self.switch('loss_gmm', 'loss_occ')

    @property
    def epsilon_low(self):
        return self.switch(-1.9, .13)

    @property
    def epsilon_high(self):
        return self.switch(-1.5, .2)

    def train_iter_low(self, labels, samples_sdf, samples_gmm):
        z = self.embeddings(torch.zeros(1, device=self.device, dtype=torch.int64))
        loss_reg = occ_loss.reg_z_loss(z)
        _, gmms = self.model.decomposition_control(z)
        loss_gmm = self.gmm_criterion(gmms, samples_gmm)
        self.logger.stash_iter('loss_gmm', loss_gmm)
        return loss_gmm + self.opt.reg_weight * loss_reg

    def train_iter_high(self, labels: T, samples_sdf: T, samples_gmm: T):
        z_d = self.mid_embeddings(torch.zeros(1, device=self.device, dtype=torch.int64)).view(1, self.opt.num_gaussians, -1)
        out, gmms = self.model.forward_mid(samples_sdf, z_d)
        loss_occ = self.criterion(out, labels)
        loss_gmm = self.gmm_criterion(gmms, samples_gmm)
        self.logger.stash_iter('loss_occ', loss_occ, 'loss_gmm', loss_gmm)
        loss = loss_occ + self.opt.gmm_weight * loss_gmm
        return loss

    def train_iter(self, labels, samples_sdf, samples_gmm):
        if self.projection_type is ProjectionType.LowProjection:
            loss = self.train_iter_low(labels, samples_sdf, samples_gmm)
        elif self.projection_type is ProjectionType.HighProjection:
            loss = self.train_iter_high(labels, samples_sdf, samples_gmm)
        else:
            raise NotImplementedError
        self.warm_up_scheduler.step()
        return loss

    @models_utils.torch_no_grad
    def switch_embedding(self):
        item = torch.zeros(1, device=self.device, dtype=torch.int64)
        self.projection_type = ProjectionType.HighProjection
        mid_embedding = torch.zeros_like(self.mid_embeddings.weight.data)
        z = self.embeddings(item)
        zs = self.model.decomposition_control.forward_low(z)[0]
        mid_embedding[item] = zs.reshape(1, -1).detach()
        self.mid_embeddings.weight.data = mid_embedding
        self.last_loss = 123454321.
        self.optimizer = Optimizer(self.mid_embeddings.parameters(), lr=1e-7)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, .5)
        self.warm_up_scheduler = train_utils.LinearWarmupScheduler(self.optimizer, 1e-3, 100)

    def early_stop(self, log, epoch) -> bool:
        loss = log[self.loss_key]
        if loss < self.epsilon_low:
            return True
        delta = self.last_loss - loss
        if delta < 1e-3 and epoch > 10 and loss < self.epsilon_high:
            return True
        self.last_loss = loss
        if (epoch + 1) % 20 == 0:
            self.scheduler.step()
            # if self.projection_type is ProjectionType.HighProjection:
                # self.prune()
        return False

    @models_utils.torch_no_grad
    def save_projection(self, res=256, verbose=False):
        z_d = self.mid_embeddings(torch.zeros(1, device=self.device, dtype=torch.int64)).view(1, self.opt.num_gaussians, -1)
        zh_base, gmms = self.model.decomposition_control.forward_mid(z_d)
        zh, attn_b = self.model.merge_zh(zh_base, gmms)
        numbers = self.get_new_ids(self.folder_name, 1)
        self.plot_occ(zh, zh_base, gmms, numbers, self.folder_name, verbose=verbose, res=res)

    def invert(self, num_epochs: int):
        for i in range(num_epochs // 2):
            if self.early_stop(self.train_epoch(i), i):
                break
        self.switch_embedding()
        for i in range(num_epochs):
            if self.early_stop(self.train_epoch(i), i):
                break
        self.save_projection()

    def __init__(self, opt: options.Options, mesh_path: str, folder_out: str):
        self.projection_type = ProjectionType.LowProjection
        super(MeshProjectionMid, self).__init__(opt, mesh_path, folder_out)
        self.mid_embeddings = nn.Embedding(1, self.opt.num_gaussians * self.opt.dim_h).to(self.device)


def main():
    for_parser = {'--model_name': {'default': 'chairs_large', 'type': str},
                  '--output_name': {'default': 'samples', 'type': str},
                  '--mesh_path': {'default': './assets/mesh/example.obj', 'type': str},
                  '--source':  {'default': 'inversion', 'type': str, 'options': ('inversion', 'random', 'training')},
                  '--num_samples':  {'default': 10, 'type': int, 'help': 'relevant for random or training'}}
    args = myparse.parse(for_parser)
    opt = options.Options(tag=args['model_name'])

    if args['source'] == 'inversion':
        model = MeshProjectionMid(opt, args['mesh_path'], args['output_name'])
        model.invert(150)
    else:
        model = occ_inference.Inference(opt)
        if args['source'] == 'random':
            model.random_plot(args['output_name'], args['num_samples'])
        else:
            model.plot(args['output_name'], args['num_samples'])
    return 0


if __name__ == '__main__':
    exit(main())
