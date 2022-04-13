import os
from custom_types import *
import multiprocessing as mp
from multiprocessing import synchronize
import options
import constants
from ui.occ_inference import Inference
from utils import files_utils
import ctypes
if constants.IS_WINDOWS or 'DISPLAY' in os.environ:
    from pynput.keyboard import Key, Controller
else:
    from ui.mock_keyboard import Key, Controller


class UiStatus(Enum):
    Waiting = 0
    GetMesh = 1
    SetGMM = 2
    SetMesh = 3
    ReplaceMesh = 4
    Exit = 5


def value_eq(value: mp.Value, status: UiStatus) -> bool:
    return value.value == status.value


def value_neq(value: mp.Value, status: UiStatus) -> bool:
    return value.value != status.value


def set_value(value: mp.Value, status: UiStatus):
    with value.get_lock():
        value.value = status.value
    print({0: 'Waiting', 1: 'GetMesh', 2: 'SetGMM', 3: 'SetMesh', 4: 'ReplaceMesh', 5: 'Exit'}[status.value])


def set_value_if_eq(value: mp.Value, status: UiStatus, check: UiStatus):
    if value_eq(value, check):
        set_value(value, status)


def set_value_if_neq(value: mp.Value, status: UiStatus, check: UiStatus):
    if value_neq(value, check):
        set_value(value, status)


def store_mesh(mesh: T_Mesh, shared_meta: mp.Array, shared_vs: mp.Array, shared_faces: mp.Array):

    def store_tensor(tensor: T, s_array: mp.Array, dtype, meta_index):
        nonlocal shared_meta_
        s_array_ = to_np_arr(s_array, dtype)
        array_ = tensor.detach().cpu().flatten().numpy()
        arr_size = array_.shape[0]
        s_array_[:array_.shape[0]] = array_
        shared_meta_[meta_index] = arr_size

    if mesh is not None:
        shared_meta_ = to_np_arr(shared_meta, np.int32)
        vs, faces = mesh
        store_tensor(vs, shared_vs, np.float32, 0)
        store_tensor(faces, shared_faces, np.int32, 1)


def load_mesh(shared_meta: mp.Array, shared_vs: mp.Array, shared_faces: mp.Array) -> V_Mesh:

    def load_array(s_array: mp.Array, dtype, meta_index) -> ARRAY:
        nonlocal shared_meta_
        s_array_ = to_np_arr(s_array, dtype)
        array_ = s_array_[: shared_meta_[meta_index]].copy()
        array_ = array_.reshape((-1, 3))
        return array_

    shared_meta_ = to_np_arr(shared_meta, np.int32)
    vs = load_array(shared_vs, np.float32, 0)
    faces = load_array(shared_faces, np.int32, 1)
    return vs, faces


def store_gmm(shared_gmm: mp.Array, gmm: TS, included: T, res: int):
    shared_arr = to_np_arr(shared_gmm, np.float32)
    mu, p, phi, eigen = gmm
    num_gaussians = included.shape[0]
    ptr = 0
    for i, (item, skip) in enumerate(zip((included, mu, p, phi, eigen), InferenceProcess.skips)):
        item = item.flatten().detach().cpu().numpy()
        if item.dtype != np.float32:
            item = item.astype(np.float32)
        shared_arr[ptr: ptr + skip * num_gaussians] = item
        if i == 0:
            shared_arr[ptr + skip * num_gaussians: ptr + skip * constants.MAX_GAUSIANS] = -1
        ptr += skip * constants.MAX_GAUSIANS
    shared_arr[-1] = float(res)


def load_gmm(shared_gmm: mp.Array) -> Tuple[TS, T, int]:
    shared_arr = to_np_arr(shared_gmm, np.float32)
    parsed_arr = []
    num_gaussians = 0
    ptr = 0
    shape = {1: (1, 1, -1), 2: (-1, 2), 3: (1, 1, -1, 3), 9: (1, 1, -1, 3, 3)}
    for i, skip in enumerate(InferenceProcess.skips):
        raw_arr = shared_arr[ptr: ptr + skip * constants.MAX_GAUSIANS]
        if i == 0:
            arr = torch.tensor([int(item) for item in raw_arr if item >= 0], dtype=torch.int64)
            num_gaussians = arr.shape[0] // 2
        else:
            arr = torch.from_numpy(raw_arr[: skip * num_gaussians]).float()
        arr = arr.view(*shape[skip])
        parsed_arr.append(arr)
        ptr += skip * constants.MAX_GAUSIANS
    return parsed_arr[1:], parsed_arr[0], int(shared_arr[-1])


def inference_process(opt: options.Options, wake_condition: synchronize.Condition,
                      sleep__condition: synchronize.Condition, status: mp.Value, samples_root: str,
                      shared_gmm: mp.Array, shared_meta: mp.Array, shared_vs: mp.Array, shared_faces: mp.Array):
    model = Inference(opt)
    items = files_utils.collect(samples_root, '.pkl')
    items = [files_utils.load_pickle(''.join(item)) for item in items]
    items = torch.stack(items, dim=0)
    # items = [int(item[1]) for item in items]
    model.set_items(items)
    keyboard = Controller()
    while value_neq(status, UiStatus.Exit):
        while value_eq(status, UiStatus.Waiting):
            with sleep__condition:
                sleep__condition.wait()
        if value_eq(status, UiStatus.GetMesh):
            set_value(status, UiStatus.SetGMM)
            gmm_info = load_gmm(shared_gmm)
            set_value_if_eq(status, UiStatus.SetMesh, UiStatus.SetGMM)
            mesh = model.get_mesh_from_mid(*gmm_info)
            if mesh is not None:
                store_mesh(mesh, shared_meta, shared_vs, shared_faces)
                keyboard.press(Key.ctrl_l)
                keyboard.release(Key.ctrl_l)
            set_value_if_eq(status, UiStatus.ReplaceMesh, UiStatus.SetMesh)
            # with wake_condition:
            #     wake_condition.notify_all()
    with wake_condition:
        wake_condition.notify_all()
    return 0


def to_np_arr(shared_arr: mp.Array, dtype) -> ARRAY:
    return np.frombuffer(shared_arr.get_obj(), dtype=dtype)


class InferenceProcess:

    skips = (2, 3, 9, 1, 3)

    def exit(self):
        set_value(self.status, UiStatus.Exit)
        with self.wake_condition:
            self.wake_condition.notify_all()
        self.model_process.join()

    def replace_mesh(self):
        mesh = load_mesh(self.shared_meta, self.shared_vs, self.shared_faces)
        self.fill_ui_mesh(mesh)
        set_value_if_eq(self.status, UiStatus.Waiting, UiStatus.ReplaceMesh)

    def get_mesh(self, res: int):
        if value_neq(self.status, UiStatus.SetGMM):
            gmms, included = self.request_gmm()
            store_gmm(self.shared_gmm, gmms, included, res)
            set_value_if_neq(self.status, UiStatus.GetMesh, UiStatus.GetMesh)
            # if value_eq(self.status, UiStatus.Waiting):
            with self.wake_condition:
                self.wake_condition.notify_all()
        return

    def __init__(self, opt, fill_ui_mesh: Callable[[V_Mesh], None], request_gmm: Callable[[], Tuple[TS, T]],
                 samples_root: List[List[str]]):
        self.status = mp.Value('i', UiStatus.Waiting.value)
        self.request_gmm = request_gmm
        self.sleep_condition = mp.Condition()
        self.wake_condition = mp.Condition()
        self.shared_gmm = mp.Array(ctypes.c_float, constants.MAX_GAUSIANS * sum(self.skips) + 1)
        self.shared_vs = mp.Array(ctypes.c_float, constants.MAX_VS * 3)
        self.shared_faces = mp.Array(ctypes.c_int, constants.MAX_VS * 8)
        self.shared_meta = mp.Array(ctypes.c_int, 2)
        self.model_process = mp.Process(target=inference_process,
                                        args=(opt, self.sleep_condition, self.wake_condition, self.status, samples_root,
                                              self.shared_gmm, self.shared_meta, self.shared_vs, self.shared_faces))
        self.fill_ui_mesh = fill_ui_mesh
        self.model_process.start()
