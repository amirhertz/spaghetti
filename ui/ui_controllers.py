from __future__ import annotations
import constants
import vtk
import vtk.util.numpy_support as numpy_support
from custom_types import *
from utils import files_utils, rotation_utils
from models import gm_utils
from ui import ui_utils, inference_processing, gaussian_status
import options


def filter_by_inclusion(gaussian: gaussian_status.GaussianStatus) -> bool:
    return gaussian.included


def filter_by_selection(gaussian: gaussian_status.GaussianStatus) -> bool:
    return gaussian.is_selected


class GmmMeshStage:

    def turn_off_selected(self):
        if self.selected is not None:
            # self.arrows.turn_off()
            self.toggle_selection(self.selected)
            self.selected = None

    def turn_gmm_off(self):
        self.turn_off_selected()
        for gaussian in self.gmm:
            gaussian.turn_off()

    def turn_gmm_on(self):
        for gaussian in self.gmm:
            gaussian.turn_on()

    def event_manger(self, object_id: str):
        if object_id in self.addresses_dict:
            return self.toggle_selection(object_id)
        elif self.arrows.check_event(object_id):
            transform = self.arrows.get_transform(object_id)
            self.update_gmm(*transform)
            return True
        return False

    def toggle_selection(self, object_id: str):
        self.gmm[self.addresses_dict[object_id]].toggle_selection()
        if self.selected is None:
            self.selected = object_id
        elif self.selected == object_id and self.gmm[self.addresses_dict[object_id]].is_not_selected:
            self.selected = None
        else:
            self.gmm[self.addresses_dict[self.selected]].toggle_selection()
            self.selected = object_id
        # if self.selected is not None:
        #     self.arrows.update_arrows_transform(self.gmm[self.addresses_dict[self.selected]])
        # else:
        #     self.arrows.turn_off()
        return True

    def toggle_inclusion_by_id(self, g_id: int, select: Optional[bool] = None) -> Tuple[bool, List[gaussian_status.GaussianStatus]]:
        toggled = []
        self.gmm[g_id].toggle_inclusion(select)
        toggled.append(self.gmm[g_id])
        if self.symmetric_mode:
            if self.gmm[g_id].twin is not None and self.gmm[g_id].twin.included != self.gmm[g_id].included:
                self.gmm[g_id].twin.toggle_inclusion(select)
                toggled.append(self.gmm[g_id].twin)
        return True, toggled

    def toggle_inclusion(self, object_id: str) -> Tuple[bool, List[gaussian_status.GaussianStatus]]:
        if object_id in self.addresses_dict:
            return self.toggle_inclusion_by_id(self.addresses_dict[object_id])
        return False, []

    def toggle_all(self):
        for gaussian in self.gmm:
            gaussian.toggle_inclusion()

    def __len__(self):
        return len(self.gmm)

    def set_opacity(self, opacity: float):
        self.view_style.opacity = opacity
        for gaussian in self.gmm:
            gaussian.set_color()

    def update_gmm(self, button: ui_utils.Buttons, key: str) -> bool:
        if self.selected is not None:
            g_id = self.addresses_dict[self.selected]
            self.gmm[g_id].apply_affine(button, key)
            if self.symmetric_mode:
                if self.gmm[g_id].twin is not None:
                    self.gmm[g_id].twin.make_symmetric(False)
            # self.arrows.update_arrows_transform(self.gmm[self.addresses_dict[self.selected]])
            return True
        return False

    def get_gmm(self) -> Tuple[TS, T]:
        raw_gmm = [g.get_raw_data() for g in self.gmm if g.included]
        phi = torch.tensor([g[0] for g in raw_gmm], dtype=torch.float32).view(1, 1, -1)
        # phi = torch.from_numpy(self.raw_gmm[0]).view(1, 1, -1).float()
        mu = torch.stack([torch.from_numpy(g[1]).float() for g in raw_gmm], dim=0).view(1, 1, -1, 3)
        p = torch.stack([torch.from_numpy(g[3]).float() for g in raw_gmm], dim=0).view(1, 1, -1, 3, 3)
        eigen = torch.stack([torch.from_numpy(g[2]).float() for g in raw_gmm], dim=0).view(1, 1, -1, 3)
        gmm = mu, p, phi, eigen
        included = torch.tensor([g.gaussian_id for g in self.gmm if g.included], dtype=torch.int64)
        return gmm, included

    def reset(self):
        for g in self.gmm:
            g.reset()
        # self.turn_off_selected()

    def remove_all(self):
        self.remove_gaussians(list(self.addresses_dict.keys()))
        self.addresses_dict = {}
        self.gmm = []

    # def switch_arrows(self, arrow_type: ui_utils.Buttons):
    #     if self.arrows.switch_arrows(arrow_type) and self.selected is not None:
    #         self.arrows.update_arrows_transform(self.gmm[self.addresses_dict[self.selected]])

    def toggle_symmetric(self, force_include: bool):
        self.symmetric_mode = not self.symmetric_mode and False
        # visited = set()
        if self.symmetric_mode:
            for i in range(len(self)):
                self.gmm[i].make_symmetric(force_include)

    def remove_gaussians(self, addresses: List[str]):
        for address in addresses:
            gaussian_id: int = self.addresses_dict[address]
            gaussian = self.gmm[gaussian_id]
            # if gaussian.is_selected:
            #     self.toggle_selection(address)
            self.gmm[gaussian_id] = None
            gaussian.delete(self.render)
            del self.addresses_dict[address]
        self.gmm = [gaussian for gaussian in self.gmm if gaussian is not None]
        self.addresses_dict = {self.gmm[i].get_address(): i for i in range(len(self.gmm))}

    def add_gaussians(self, gaussians: List[gaussian_status.GaussianStatus]) -> List[str]:
        new_addresses = []
        for i, gaussian in enumerate(gaussians):
            gaussian_copy = gaussian.copy(self.render, self.view_style, is_selected=False)
            self.gmm.append(gaussian_copy)
            new_addresses.append(gaussian_copy.get_address())
        self.addresses_dict = {self.gmm[i].get_address(): i for i in range(len(self.gmm))}
        return new_addresses

    def make_twins(self, address_a: str, address_b: str):
        if address_a in self.addresses_dict and address_b in self.addresses_dict:
            gaussian_a, gaussian_b = self.gmm[self.addresses_dict[address_a]], self.gmm[self.addresses_dict[address_b]]
            gaussian_a.twin = gaussian_b
            gaussian_b.twin = gaussian_a

    def split_mesh_by_gmm(self, mesh) -> Dict[int, T]:
        faces_split = {}
        mu, p, phi, _ = self.get_gmm()[0]
        eigen = torch.stack([torch.from_numpy(g.get_view_eigen()).float() for g in self.gmm if g.included], dim=0).view(1, 1, -1, 3)
        gmm = mu, p, phi, eigen
        faces_split_ = gm_utils.split_mesh_by_gmm(mesh, gmm)
        counter = 0
        for i in range(len(self.gmm)):
            if self.gmm[i].disabled:
                faces_split[i] = None
            else:
                faces_split[i] = faces_split_[counter]
                counter += 1
        return faces_split

    @staticmethod
    def get_part_face(mesh: V_Mesh, faces_inds: T) -> Tuple[T_Mesh, T]:
        mesh = mesh[0], torch.from_numpy(mesh[1]).long()
        mask = faces_inds.ne(0)
        faces = mesh[1][mask]
        vs_inds = faces.flatten().unique()
        vs = mesh[0][vs_inds]
        mapper = torch.zeros(mesh[0].shape[0], dtype=torch.int64)
        mapper[vs_inds] = torch.arange(vs.shape[0])
        return (vs, mapper[faces]), faces_inds[mask]

    def save(self, root: str, filter_faces: Callable[[gaussian_status.GaussianStatus], bool] = filter_by_inclusion):
        if self.faces is not None:
            if self.gmm_id == -1:
                name = "mix"
            else:
                name = str(self.gmm_id)
            path = f"{root}/{files_utils.get_time_name(name)}"
            faces = list(filter(lambda x: x[1] is not None, self.faces.items()))
            mesh = self.vs, np.concatenate(list(map(lambda x: x[1], faces)))
            faces_inds = map(lambda x:
                             torch.ones(x[1].shape[0], dtype=torch.int64)
                             if filter_faces(self.gmm[x[0]]) else torch.zeros(x[1].shape[0], dtype=torch.int64), faces)
            faces_inds = torch.cat(list(faces_inds))
            # if name != 'mix':
            #     mesh, faces_inds = self.get_part_face(mesh, faces_inds)
            files_utils.export_mesh(mesh, path)
            files_utils.export_list(faces_inds.tolist(), f"{path}_faces")

    def aggregate_symmetric(self) -> Dict[str, int]:
        if not self.symmetric_mode:
            return self.votes
        out = {}
        for item in self.votes:
            actor_id = self.addresses_dict[item]
            twin = self.gmm[actor_id].twin
            out[item] = self.votes[item]
            if twin is not None and twin.get_address() not in self.votes:
                out[twin.get_address()] = self.votes[item]
        return out

    def aggregate_votes(self) -> List[int]:
        # to_do = self.add_selection if select else self.clear_selection
        actors_id = []
        # votes = self.aggregate_symmetric()
        for item in self.votes:
            actor_id = self.addresses_dict[item]
            actors_id.append(actor_id)
        self.votes = {}
        return actors_id

    def vote(self, *actors: Optional[vtk.vtkActor]):
        for actor in actors:
            if actor is not None:
                address = actor.GetAddressAsString('')
                if address in self.addresses_dict:
                    if address not in self.votes:
                        self.votes[address] = 0
                    self.votes[address] += 1

    @staticmethod
    def faces_to_vtk_faces(faces: Union[T, ARRAY]):
        if type(faces) is T:
            faces = faces.detach().cpu().numpy()
        cells_npy = np.column_stack(
            [np.full(faces.shape[0], 3, dtype=np.int64), faces.astype(np.int64)]).ravel()
        faces_vtk = vtk.vtkCellArray()
        faces_vtk.SetCells(faces.shape[0], numpy_support.numpy_to_vtkIdTypeArray(cells_npy))
        return faces_vtk

    def get_mesh_part(self, vs: vtk.vtkPoints, faces: Optional[Union[T, ARRAY]]) -> Optional[vtk.vtkPolyData]:
        if faces is not None:
            # actor_mesh = vtk.vtkActor()
            mesh = vtk.vtkPolyData()
            # mapper = vtk.vtkPolyDataMapper()
            mesh.SetPoints(vs)
            mesh.SetPolys(self.faces_to_vtk_faces(faces))
            # mapper.SetInputData(mesh)
            # actor_mesh.SetMapper(mapper)
            # actor_mesh.GetProperty().SetOpacity(0.3)
            # actor_mesh.PickableOff()
            # if self.to_init:
            #     self.render.AddActor(actor_mesh)
            return mesh
        return None

    def add_gmm(self) -> List[gaussian_status.GaussianStatus]:
        gmms = []
        if len(self.raw_gmm) > 0:
            phi = self.raw_gmm[0]
            phi = np.exp(phi)
            phi = phi / phi.sum()
            for i, gaussian in enumerate(zip(*self.raw_gmm)):
                gaussian = gaussian_status.GaussianStatus(gaussian, (self.gmm_id, i), False, self.view_style,
                                                          self.render, phi[i])
                gmms.append(gaussian)
        return gmms

    def add_mesh(self, base_mesh: T_Mesh, split_mesh: bool = True, for_slider: bool = True):
        if base_mesh is not None:
            vs_vtk = vtk.vtkPoints()
            self.vs = base_mesh[0]
            if for_slider:
                vs_ui = self.init_mesh_pos(base_mesh[0])
            else:
                vs_ui = self.vs
            vs_vtk.SetData(numpy_support.numpy_to_vtk(vs_ui.numpy()))
            if split_mesh:
                self.faces = self.split_mesh_by_gmm(base_mesh)
                for i in range(len(self.gmm)):
                    part_mesh = self.get_mesh_part(vs_vtk, self.faces[i])
                    self.gmm[i].replace_part(part_mesh)
            else:
                part_mesh = self.get_mesh_part(vs_vtk, base_mesh[1])
                self.gmm[0].replace_part(part_mesh)

    def set_brush(self, is_draw: bool):
        self.render.set_brush(is_draw)

    def replace_mesh(self, mesh: Optional[V_Mesh]):
        mesh = torch.from_numpy(mesh[0]).float(), torch.from_numpy(mesh[1]).long()
        self.add_mesh(mesh, for_slider=False)
        # if mesh is None:
        #     return
        # else:
        #     reduction = 1 - 50000. / mesh[1].shape[0]
        #     source_ = MeshStage.mesh_to_polydata(mesh)
        #     source_ = MeshStage.smooth_mesh(source_, ui_utils.SmoothingMethod.Taubin)
        #     self.decimate_mesh(source_, reduction, out=self.mapper.GetInput())
        # self.is_changed = True
        # if not self.to_init:
        #     self.to_init = True
        #     self.render.AddActor(self.actor)

    def init_mesh_pos(self, vs: T):
        vs = vs.clone()
        r_a = rotation_utils.get_rotation_matrix(150, 1, degree=True)
        r_b = rotation_utils.get_rotation_matrix(-15, 0, degree=True)
        r = torch.from_numpy(np.einsum('km,mn->kn', r_b, r_a)).float()
        vs = torch.einsum('ad,nd->na', r, vs)
        vs[:, 0] += self.gmm_id * 2
        return vs

    @staticmethod
    def mesh_to_polydata(mesh: Union[T_Mesh, V_Mesh], source: Optional[vtk.vtkPolyData] = None) -> vtk.vtkPolyData:
        if source is None:
            source = vtk.vtkPolyData()
        vs, faces = mesh
        if type(vs) is T:
            vs, faces = vs.detach().cpu().numpy(), faces.detach().cpu().numpy()
        new_vs_vtk = numpy_support.numpy_to_vtk(vs)
        cells_npy = np.column_stack(
            [np.full(faces.shape[0], 3, dtype=np.int64), faces.astype(np.int64)]).ravel()
        vs_vtk, faces_vtk = vtk.vtkPoints(), vtk.vtkCellArray()
        vs_vtk.SetData(new_vs_vtk)
        faces_vtk.SetCells(faces.shape[0], numpy_support.numpy_to_vtkIdTypeArray(cells_npy))
        source.SetPoints(vs_vtk)
        source.SetPolys(faces_vtk)
        return source

    @property
    def included(self):
        for g in self.gmm:
            if g.included:
                return True
        return False

    def move_mesh_to_end(self, cycle: int):
        self.offset += cycle
        vs = None
        for i in range(len(self)):
            mapper = self.gmm[i].mapper
            if mapper is not None and mapper.GetInput() is not None:
                vs_vtk = mapper.GetInput().GetPoints()
                if vs is None:
                    vs = numpy_support.vtk_to_numpy(vs_vtk.GetData())
                    vs[:, 0] += cycle * 2
                vs_vtk.SetData(numpy_support.numpy_to_vtk(vs))

    def pick(self, actor_address: str) -> bool:
        return actor_address in self.addresses_dict

    def __init__(self, opt: options.Options, shape_path: List[str], render: ui_utils.CanvasRender, render_number: int,
                 view_style: ui_utils.ViewStyle, to_init=True):
        self.view_style = view_style
        self.votes = {}
        self.shape_id = shape_path[1]
        self.gmm_id = render_number
        self.render = render
        self.symmetric_mode = sum(opt.symmetric) > 0 and False
        self.selected = None
        self.offset = render_number
        # self.arrows = arrows.ArrowManger(render)
        if self.shape_id != '-1':
            self.base_mesh = files_utils.load_mesh( ''.join(shape_path))
            self.raw_gmm = files_utils.load_gmm(f'{shape_path[0]}/{shape_path[1]}.txt', as_np=True)[:-1]
        else:
            self.base_mesh = None
            self.raw_gmm = []
        self.to_init = to_init
        self.is_changed = False
        self.gmm: List[gaussian_status.GaussianStatus] = self.add_gmm()
        self.vs = self.faces = None
        self.add_mesh(self.base_mesh)
        self.addresses_dict: Dict[str, int] = {self.gmm[i].get_address(): i for i in range(len(self.gmm))}
        if self.symmetric_mode:
            for i in range(len(self) // 2):
                self.make_twins(self.gmm[i].get_address(), self.gmm[i + len(self) // 2].get_address())
        self.toggle_all()
        # if self.raw_gmm:
        #     gmms = self.get_gmm()[0]
        #     files_utils.export_gmm(gmms, 0, f"./{render_number}")


class GmmStatuses:

    def __len__(self):
        return len(self.gmms)

    def switch_arrows(self, arrow_type: ui_utils.Buttons):
        self.main_gmm.switch_arrows(arrow_type)

    def turn_gmm_off(self):
        self.main_gmm.turn_gmm_off()

    def turn_gmm_on(self):
        self.main_gmm.turn_gmm_on()

    def update_gmm(self, button: ui_utils.Buttons, key: str) -> bool:
        return self.main_gmm.update_gmm(button, key)

    def toggle_symmetric(self, force_include: bool = False):
        for gmm in self.gmms:
            gmm.toggle_symmetric(force_include)

    def event_manger(self, object_id: str):
        for gmm in self.gmms:
            if gmm.event_manger(object_id):
                return True
        return False

    def toggle_inclusion(self, object_id: str):
        for gmm in self.gmms:
            if gmm.toggle_inclusion(object_id)[0]:
                return True
        return False

    @property
    def main_gmm(self) -> GmmMeshStage:
        return self.gmms[0]

    def reset(self):
        for gmm in self.gmms:
            gmm.reset()

    def set_brush(self, is_draw: bool):
        for gmm in self.gmms:
            gmm.set_brush(is_draw)

    def move_mesh_to_end(self, ptr: int):
        self.gmms[ptr].move_mesh_to_end(len(self))

    def pick(self,  actor_address: str) -> Optional[GmmMeshStage]:
        for gmm in self.gmms:
            if gmm.pick(actor_address):
                return gmm
        return None

    def __init__(self, opt: options.Options, shape_paths: List[List[str]], render, view_styles: List[ui_utils.ViewStyle]):
        self.gmms = [GmmMeshStage(opt, shape_path, render, i, view_style) for i, (shape_path, view_style) in
                     enumerate(zip(shape_paths, view_styles))]


def to_local(func):
    def inner(self: MeshGmmStatuses.TransitionController, mouse_pos: Optional[Tuple[int, int]], *args, **kwargs):
        if mouse_pos is not None:
            size = self.render.GetRenderWindow().GetScreenSize()
            aspect = self.render.GetAspect()
            mouse_pos = float(mouse_pos[0]) / size[0] - .5, float(mouse_pos[1]) / size[1] - .5
            mouse_pos = torch.tensor([mouse_pos[0] / aspect[1], mouse_pos[1] / aspect[0]])
        return func(self, mouse_pos, *args, **kwargs)
    return inner


class MeshGmmStatuses(GmmStatuses):

    def aggregate_votes(self, select: bool):
        if self.cur_canvas < len(self.gmms):
            stage = self.gmms[self.cur_canvas]
            changed = stage.aggregate_votes()
            changed = list(filter(lambda x: not stage.gmm[x].disabled and stage.gmm[x].is_selected != select, changed))
            for item in changed:
                stage.gmm[item].toggle_selection()
            return len(changed) > 0

    def vote(self, *actors: Optional[vtk.vtkActor]):
        self.gmms[self.cur_canvas].vote(*actors)

    def init_draw(self, side: int):
        self.cur_canvas = side

    def sort_gmms(self, gmms, included):
        order = torch.arange(gmms[0].shape[2]).tolist()
        order = sorted(order, key=lambda x: included[x][0] * 100 + included[x][1])
        gmms = [[item[:, :, order[i]] for item in gmms] for i in range(gmms[0].shape[2])]
        gmms = [torch.stack([gmms[j][i] for j in range(len(gmms))], dim=2) for i in range(len(gmms[0]))]
        return gmms

    def save_light(self, root, gmms):
        gmms = self.sort_gmms(*gmms)
        save_dict = {'ids': {
                             gmm.shape_id: [gaussian.gaussian_id[1] for gaussian in gmm.gmm if gaussian.included]
                             for gmm in self.gmms if gmm.included},
                     'gmm': gmms}
        path = f"{root}/{files_utils.get_time_name('light')}"
        files_utils.save_pickle(save_dict, path)

    def save(self, root: str, gmms):
        # for gmm in self.gmms:
        #     if gmm.included:
        #         gmm.save(root)
        if len(gmms[0]) > 0:
            self.save_light(root, gmms)

    def set_brush(self, is_draw: bool):
        super(MeshGmmStatuses, self).set_brush(is_draw)
        self.main_gmm.render.set_brush(is_draw)

    def update_mesh(self, res=128):
        if self.model_process is not None:
            self.model_process.get_mesh(res)
            return True
        return False
        # self.all_info[side] = gaussian_inds

    def request_gmm(self) -> Tuple[TS, T]:
        gmm, included = self.main_gmm.get_gmm()
        return gmm, included

    def replace_mesh(self):
        if self.model_process is not None:
            self.model_process.replace_mesh()

    def exit(self):
        if self.model_process is not None:
            self.model_process.exit()

    @property
    def main_stage(self) -> GmmMeshStage:
        return self.gmms[0]

    @property
    def stages(self):
        return self.gmms

    class TransitionController:

        @property
        def moving_axis(self) -> int:
            return {ui_utils.EditDirection.X_Axis: 0,
                    ui_utils.EditDirection.Y_Axis: 2,
                    ui_utils.EditDirection.Z_Axis: 1}[self.edit_direction]

        def get_delta_translation(self, mouse_pos: T) -> ARRAY:
            delta_3d = np.zeros(3)
            axis = self.moving_axis
            vec = mouse_pos - self.origin_mouse
            delta = torch.einsum('d,d', vec, self.dir_2d[:, axis])
            delta_3d[axis] = delta
            return delta_3d

        def get_delta_rotation(self, mouse_pos: T) -> ARRAY:
            projections = []
            for pos in (self.origin_mouse, mouse_pos):
                vec = pos - self.transition_origin_2d
                projection = torch.einsum('d,da->a', vec, self.dir_2d)
                projection[self.moving_axis] = 0
                projection = nnf.normalize(projection, p=2, dim=0)
                projections.append(projection)
            sign = (projections[0][(self.moving_axis + 2) % 3] * projections[1][(self.moving_axis + 1) % 3]
                    - projections[0][(self.moving_axis + 1) % 3] * projections[1][(self.moving_axis + 2) % 3] ).sign()
            angle = (torch.acos(torch.einsum('d,d', *projections)) * sign).item()
            return ui_utils.get_rotation_matrix(angle, self.moving_axis)

        def get_delta_scaling(self, mouse_pos: T) -> ARRAY:
            raise NotImplementedError

        def toggle_edit_direction(self, direction: ui_utils.EditDirection):
            self.edit_direction = direction

        @to_local
        def get_transition(self, mouse_pos: Optional[T]) -> ui_utils.Transition:
            transition = ui_utils.Transition(self.transition_origin.numpy(), self.transition_type)
            if mouse_pos is not None:
                if self.transition_type is ui_utils.EditType.Translating:
                    transition.translation = self.get_delta_translation(mouse_pos)
                elif self.transition_type is ui_utils.EditType.Rotating:
                    transition.rotation = self.get_delta_rotation(mouse_pos)
                elif self.transition_type is ui_utils.EditType.Scaling:
                    transition.rotation = self.get_delta_scaling(mouse_pos)
            return transition

        @to_local
        def init_transition(self, mouse_pos: Tuple[int, int], transition_origin: T, transition_type: ui_utils.EditType):
            transform_mat_vtk = self.camera.GetViewTransformMatrix()
            dir_2d = torch.zeros(3, 4)
            for i in range(3):
                for j in range(4):
                    dir_2d[i, j] = transform_mat_vtk.GetElement(i, j)
            self.transition_origin = transition_origin
            transition_origin = torch.tensor(transition_origin.tolist() + [1])
            transition_origin_2d = torch.einsum('ab,b->a', dir_2d, transition_origin)
            self.transition_origin_2d = transition_origin_2d[:2] / transition_origin_2d[-1].abs()
            # print(f"<{self.transition_origin[0]}, {self.transition_origin[1]}>")
            # print(mouse_pos)
            self.origin_mouse, self.dir_2d = mouse_pos,  nnf.normalize(dir_2d[:2, :3], p=2, dim=1)
            self.transition_type = transition_type

        @property
        def camera(self):
            return self.render.GetActiveCamera()

        def __init__(self, render: ui_utils.CanvasRender):
            self.render = render
            self.transition_origin = torch.zeros(3)
            self.transition_origin_2d = torch.zeros(2)
            self.origin_mouse, self.dir_2d = torch.zeros(2), torch.zeros(2, 3)
            self.edit_direction = ui_utils.EditDirection.X_Axis
            self.transition_type = ui_utils.EditType.Translating

    @property
    def selected_gaussians(self) -> Iterable[gaussian_status.GaussianStatus]:
        return filter(lambda x: x.is_selected, self.main_stage.gmm)

    def temporary_transition(self, mouse_pos: Optional[Tuple[int, int]] = None, end=False) -> bool:
        transition = self.transition_controller.get_transition(mouse_pos)
        is_change = False
        for gaussian in self.selected_gaussians:
            if end:
                is_change = gaussian.end_transition(transition) or is_change
            else:
                is_change = gaussian.temporary_transition(transition) or is_change
        return is_change

    def end_transition(self, mouse_pos: Optional[Tuple[int, int]]) -> bool:
        return self.temporary_transition(mouse_pos, True)

    def init_transition(self, mouse_pos, transition_type: ui_utils.EditType):
        center = list(map(lambda x: x.mu_baked, self.selected_gaussians))
        if len(center) == 0:
            return
        # center = torch.from_numpy(np.stack(center, axis=0).mean(0))
        center = torch.zeros(3)
        self.transition_controller.init_transition(mouse_pos, center, transition_type)

    def toggle_edit_direction(self, direction: ui_utils.EditDirection):
        self.transition_controller.toggle_edit_direction(direction)

    def clear_selection(self) -> bool:
        is_changed = False
        for gaussian in self.selected_gaussians:
            gaussian.toggle_selection()
            is_changed = True
        return is_changed

    def __init__(self, opt: options.Options, shape_paths: List[List[str]], render, view_styles: List[ui_utils.ViewStyle],
                 with_model: bool):
        super(MeshGmmStatuses, self).__init__(opt, shape_paths, render, view_styles)
        if with_model:
            self.model_process = inference_processing.InferenceProcess(opt, self.main_stage.replace_mesh,
                                                                       self.request_gmm,
                                                                       shape_paths)
        else:
            self.model_process = None
        self.counter = 0
        self.cur_canvas = 0
        self.transition_controller = MeshGmmStatuses.TransitionController(self.main_stage.render)


class MeshGmmUnited(MeshGmmStatuses):

    def save(self, root: str):
        super(MeshGmmUnited, self).save(root)
        self.main_gmm.save(root, filter_by_selection)

    def aggregate_votes(self, select: bool):
        if self.cur_canvas < len(self.gmms):
            stage = self.gmms[self.cur_canvas]
            changed = stage.aggregate_votes()
            changed = list(filter(lambda x: not stage.gmm[x].disabled and stage.gmm[x].included != select, changed))
            for item in changed:
                is_toggled, toggled = stage.toggle_inclusion_by_id(item, select)
                if is_toggled:
                    if toggled[0].included:
                        new_addresses = self.main_gmm.add_gaussians(toggled)
                        for gaussian, new_address in zip(toggled, new_addresses):
                            self.stage_mapper[gaussian.get_address()] = new_address
                        self.make_twins(toggled, new_addresses)
                    else:
                        addresses = [gaussian.get_address() for gaussian in toggled]
                        addresses = list(filter(lambda x: x in self.stage_mapper, addresses))
                        self.main_gmm.remove_gaussians([self.stage_mapper[address] for address in addresses])
                        for address in addresses:
                            del self.stage_mapper[address]
            return len(changed) > 0
        else:
            return self.update_selection(select)

    def update_selection(self, select: bool):
        changed = self.main_stage.aggregate_votes()
        changed = filter(lambda x: self.main_stage.gmm[x].is_selected != select, changed)
        for item in changed:
            self.main_stage.gmm[item].toggle_selection()
        return False

    def vote(self, *actors: Optional[vtk.vtkActor]):
        if self.cur_canvas < len(self.gmms):
            self.gmms[self.cur_canvas].vote(*actors)
        else:
            self.main_gmm.vote(*actors)

    def reset(self):
        super(MeshGmmUnited, self).reset()
        self.main_gmm.remove_all()
        for gmm in self.gmms:
            gmm.toggle_all()
        self.stage_mapper = {}

    def event_manger(self, object_id: str):
        return self.toggle_inclusion(object_id) or self.main_gmm.event_manger(object_id)

    def make_twins(self, toggled: List[gaussian_status.GaussianStatus], new_addresses : List[str]):
        if len(new_addresses) == 2:
            self.main_gmm.make_twins(*new_addresses)
        else:
            if toggled[0].twin is not None and toggled[0].twin.get_address() in self.stage_mapper:
                self.main_gmm.make_twins(new_addresses[0], self.stage_mapper[toggled[0].twin.get_address()])

    def toggle_symmetric(self, force_include: bool = False):
        super(MeshGmmUnited, self).toggle_symmetric(force_include)
        self.main_gmm.toggle_symmetric(force_include)

    @property
    def main_gmm(self) -> GmmMeshStage:
        return self.main_gmm_

    @property
    def main_stage(self) -> GmmMeshStage:
        return self.main_gmm_

    def __init__(self, opt: options.Options, gmm_paths: List[int], renders_right, view_styles: List[ui_utils.ViewStyle],
                 main_render: ui_utils.CanvasRender, with_model: bool):
        self.main_gmm_ = GmmMeshStage(opt, -1, main_render, len(gmm_paths), view_styles[-1], to_init=False)
        super(MeshGmmUnited, self).__init__(opt, gmm_paths, renders_right, view_styles[:-1], with_model)
        self.main_render = main_render
        self.reset()
        self.stage_mapper: Dict[str, str] = {}


def main():
    opt = options.Options(tag="chairs_sym_hard").load()
    model = train_utils.model_lc(opt)[0]
    model = model.to(CPU)
    colors = torch.rand(opt.num_gaussians, 3)
    shape_nums = 1103, 1637, 2954, 3631, 4814
    for shape_num in shape_nums:
        mesh = files_utils.load_mesh(f"{opt.cp_folder}/occ/samples_{shape_num}")
        gmm = files_utils.load_gmm(f"{opt.cp_folder}/gmms/samples_{shape_num}")
        vs, faces = mesh
        phi, mu, eigen, p, _ = [item.unsqueeze(0).unsqueeze(0) for item in gmm]
        gmm = mu, p, phi, eigen
        attention = model.get_attention(vs.unsqueeze(0), torch.tensor([shape_num], dtype=torch.int64))[-4:]
        # _, supports = gm_utils.hierarchical_gm_log_likelihood_loss([gmm], vs.unsqueeze(0), get_supports=True)
        # supports = supports[0][0]
        supports = torch.cat(attention, dim=0)
        supports = supports.mean(-1).mean(0)
        label = supports.argmax(1)
        colors_ = colors[label]
        files_utils.export_mesh((vs, faces), f"{constants.OUT_ROOT}/{opt.tag}_{shape_num}b", colors=colors_)
    return 0


if __name__ == '__main__':

    from utils import train_utils
    exit(main())
