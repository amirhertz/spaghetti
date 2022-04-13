from custom_types import *
import constants
from tqdm import tqdm
from utils import files_utils
import os
import options
from models import models_utils, occ_gmm


LI = Union[T, float, int]
Models = {'spaghetti': occ_gmm.Spaghetti}


def is_model_clean(model: nn.Module) -> bool:
    for wh in model.parameters():
        if torch.isnan(wh).sum() > 0:
            return False
    return True


def model_factory(opt: options.Options, override_model: Optional[str], device: D) -> models_utils.Model:
    if override_model is None:
        return Models[opt.model_name](opt).to(device)
    return Models[override_model](opt).to(device)


def load_model(opt, device, suffix: str = '', override_model: Optional[str] = None) -> models_utils.Model:
    model_path = f'{opt.cp_folder}/model{"_" + suffix if suffix else ""}'
    model = model_factory(opt, override_model, device)
    name = opt.model_name if override_model is None else override_model
    if os.path.isfile(model_path):
        print(f'loading {name} model from {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f'init {name} model')
    return model


def save_model(model, path):
    if constants.DEBUG:
        return False
    print(f'saving model in {path}')
    torch.save(model.state_dict(), path)
    return True


def model_lc(opt: options.Options, override_model: Optional[str] = None) -> Tuple[occ_gmm.Spaghetti, options.Options]:

    def save_model(model_: models_utils.Model, suffix: str = ''):
        nonlocal already_init
        if override_model is not None and suffix == '':
            suffix = override_model
        model_path = f'{opt.cp_folder}/model{"_" + suffix if suffix else ""}'
        if constants.DEBUG or 'debug' in opt.tag:
            return False
        if not already_init:
            files_utils.init_folders(model_path)
            files_utils.save_pickle(opt, params_path)
            already_init = True
        if is_model_clean(model_):
            print(f'saving {opt.model_name} model at {model_path}')
            torch.save(model_.state_dict(), model_path)
        elif os.path.isfile(model_path):
            print(f'model is corrupted')
            print(f'loading {opt.model_name} model from {model_path}')
            model.load_state_dict(torch.load(model_path, map_location=opt.device))
        return True

    already_init = False
    params_path = f'{opt.cp_folder}/options.pkl'
    opt_ = files_utils.load_pickle(params_path)

    if opt_ is not None:
        opt_.device = opt.device
        opt = opt_
        already_init = True
    model = load_model(opt, opt.device, override_model=override_model)
    model.save_model = save_model
    return model, opt


class Logger:

    def __init__(self, level: int = 0):
        self.level_dictionary = dict()
        self.iter_dictionary = dict()
        self.level = level
        self.progress: Union[N, tqdm] = None
        self.iters = 0
        self.tag = ''

    @staticmethod
    def aggregate(dictionary: dict, parent_dictionary: Union[dict, N] = None) -> dict:
        aggregate_dictionary = dict()
        for key in dictionary:
            if 'counter' not in key:
                aggregate_dictionary[key] = dictionary[key] / float(dictionary[f"{key}_counter"])
                if parent_dictionary is not None:
                    Logger.stash(parent_dictionary, (key,  aggregate_dictionary[key]))
        return aggregate_dictionary

    @staticmethod
    def flatten(items: Tuple[Union[Dict[str, LI], str, LI], ...]) -> List[Union[str, LI]]:
        flat_items = []
        for item in items:
            if type(item) is dict:
                for key, value in item.items():
                    flat_items.append(key)
                    flat_items.append(value)
            else:
                flat_items.append(item)
        return flat_items

    @staticmethod
    def stash(dictionary: Dict[str, LI], items: Tuple[Union[Dict[str, LI], str, LI], ...]) -> Dict[str, LI]:
        flat_items = Logger.flatten(items)
        for i in range(0, len(flat_items), 2):
            key, item = flat_items[i], flat_items[i + 1]
            if type(item) is T:
                item = item.item()
            if key not in dictionary:
                dictionary[key] = 0
                dictionary[f"{key}_counter"] = 0
            dictionary[key] += item
            dictionary[f"{key}_counter"] += 1
        return dictionary

    def stash_iter(self, *items: Union[Dict[str, LI], str, LI]):
        self.iter_dictionary = self.stash(self.iter_dictionary, items)
        return self

    def stash_level(self, *items: Union[Dict[str, LI], str, LI]):
        self.level_dictionary = self.stash(self.level_dictionary, items)

    def reset_iter(self, *items: Union[Dict[str, LI], str, LI]):
        if len(items) > 0:
            self.stash_iter(*items)
        aggregate_dictionary = self.aggregate(self.iter_dictionary, self.level_dictionary)
        self.progress.set_postfix(aggregate_dictionary)
        self.progress.update()
        self.iter_dictionary = dict()
        return self

    def start(self, iters: int, tag: str = ''):
        if self.progress is not None:
            self.stop()
        if iters < 0:
            iters = self.iters
        if tag == '':
            tag = self.tag
        self.iters, self.tag = iters, tag
        self.progress = tqdm(total=self.iters, desc=f'{self.tag} {self.level}')
        return self

    def stop(self, aggregate: bool = True):
        if aggregate:
            aggregate_dictionary = self.aggregate(self.level_dictionary)
            self.progress.set_postfix(aggregate_dictionary)
        self.level_dictionary = dict()
        self.progress.close()
        self.progress = None
        self.level += 1
        return aggregate_dictionary

    def reset_level(self, aggregate: bool = True):
        self.stop(aggregate)
        self.start()


class LinearWarmupScheduler:

    def get_lr(self):
        if self.cur_iter >= self.num_iters:
            return [self.target_lr] * len(self.base_lrs)
        alpha = self.cur_iter / self.num_iters
        return [base_lr + delta_lr * alpha for base_lr, delta_lr in zip(self.base_lrs, self.delta_lrs)]

    def step(self):
        if not self.finished:
            for group, lr in zip(self.optimizer.param_groups,  self.get_lr()):
                group['lr'] = lr
            self.cur_iter += 1.
            self.finished = self.cur_iter > self.num_iters

    def __init__(self, optimizer, target_lr, num_iters):
        self.cur_iter = 0.
        self.target_lr = target_lr
        self.num_iters = num_iters
        self.finished = False
        self.optimizer = optimizer
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.delta_lrs = [target_lr - base_lr for base_lr in self.base_lrs]
