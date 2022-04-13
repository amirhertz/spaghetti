from __future__ import annotations
import os
import pickle
import constants as const
from custom_types import *


class Options:

    def load(self):
        device = self.device
        if os.path.isfile(self.save_path):
            print(f'loading opitons from {self.save_path}')
            with open(self.save_path, 'rb') as f:
                options = pickle.load(f)
            options.device = device
            return options
        return self

    def save(self):
        if os.path.isdir(self.cp_folder):
            # self.already_saved = True
            with open(self.save_path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @property
    def info(self) -> str:
        return f'{self.model_name}_{self.tag}'

    @property
    def cp_folder(self):
        return f'{const.CHECKPOINTS_ROOT}{self.info}'

    @property
    def save_path(self):
        return f'{const.CHECKPOINTS_ROOT}{self.info}/options.pkl'

    def fill_args(self, args):
        for arg in args:
            if hasattr(self, arg):
                setattr(self, arg, args[arg])

    def __init__(self, **kwargs):
        self.device = CUDA(0)
        self.tag = 'airplanes'
        self.dataset_name = 'shapenet_airplanes_wm_sphere_sym_train'
        self.epochs = 2000
        self.model_name = 'spaghetti'
        self.dim_z = 256
        self.pos_dim = 256 - 3
        self.dim_h = 512
        self.dim_zh = 512
        self.num_gaussians = 16
        self.min_split = 4
        self.max_split = 12
        self.gmm_weight = 1
        self.decomposition_network = 'transformer'
        self.decomposition_num_layers = 4
        self.num_layers = 4
        self.num_heads = 4
        self.num_layers_head = 6
        self.num_heads_head = 8
        self.head_occ_size = 5
        self.head_occ_type = 'skip'
        self.batch_size = 18
        self.num_samples = 2000
        self.dataset_size = -1
        self.symmetric = (True, False, False)
        self.data_symmetric = (True, False, False)
        self.lr_decay = .9
        self.lr_decay_every = 500
        self.warm_up = 2000
        self.reg_weight = 1e-4
        self.disentanglement = True
        self.use_encoder = True
        self.disentanglement_weight = 1
        self.augmentation_rotation = 0.3
        self.augmentation_scale = .2
        self.augmentation_translation = .3
        self.fill_args(kwargs)
