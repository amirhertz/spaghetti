from __future__ import annotations
import vtk
from custom_types import *
from ui import ui_utils
import constants
import vtk.util.numpy_support as numpy_support


class GaussianData:

    def make_symmetric(self, other: GaussianData):
        # reflection = np.eye(3)
        # reflection[0, 0] = -1
        # self.total_translate = np.einsum('ad,d->a', reflection, other.total_translate)
        self.total_translate = other.total_translate.copy()
        self.total_translate[0] *= -1
        self.total_rotate = other.total_rotate.copy()
        # self.total_rotate = np.einsum('ab,bc->ac', reflection, other.total_rotate)

    @staticmethod
    def to_positive(p):
        up_dir = 1
        eye = np.eye(3)
        all_dots = (p[:, :] * eye[up_dir, None, :]).sum(-1)
        up_axis = all_dots.__abs__().argmax()
        return up_axis, all_dots[up_axis] < 0

    def permute_p(self, p):
        up_dir = 1
        p_new = np.eye(3)
        p_new[up_dir] = p[self.up_axis]
        if self.reflect_up:
            p_new[up_dir] = -p_new[up_dir]
        p_new[(up_dir + 1) % 3] = p[(self.up_axis + 1) % 3]
        p_new[(up_dir + 2) % 3, :] = np.cross(p_new[up_dir, :], p_new[(up_dir + 1) % 3, :])
        return p_new

    def rotate(self, transition: ui_utils.Transition):
        mu = self.mu_baked - transition.transition_origin
        mu = np.einsum('ab,b->a', transition.rotation, mu) + transition.transition_origin
        self.total_translate = mu - self.mu
        self.total_rotate = np.einsum('ab,bc->ac', transition.rotation, self.total_rotate)

    def stretch(self, amount):
        scale = 0.9 if amount < 0 else 1 / .9
        self.eigen = self.eigen * scale

    def translate(self, transition: ui_utils.Transition):
        self.total_translate = self.total_translate + transition.translation

    def get_view_eigen(self):
        scale = (self.mu_baked ** 2).sum() / (self.mu ** 2).sum()
        return self.eigen * scale

    def get_raw_data(self):
        # p = np.einsum('da,db->ba', self.p, self.total_rotate)
        p = np.einsum('ab,bc->ac', self.total_rotate, self.p.transpose()).transpose()
        return self.phi, self.mu_baked, self.eigen, p

    def copy_data(self):
        return [item.copy() if type(item) is ARRAY else item for item in self.get_raw_data()]

    def get_view_data(self):
        phi, mu, eigen, p = self.get_raw_data()
        p = self.permute_p(p)
        return phi, mu, eigen, p

    @property
    def mu_baked(self) -> ARRAY:
        return self.mu + self.total_translate

    @property
    def phi(self) -> float:
        return self.data[0]

    @property
    def mu(self) -> ARRAY:
        return self.data[1]

    @property
    def eigen(self) -> ARRAY:
        return self.data[2]

    @property
    def p(self) -> ARRAY:
        return self.data[3]

    @mu.setter
    def mu(self, new_mu: ARRAY):
        self.data[1] = new_mu

    @p.setter
    def p(self, new_p: ARRAY):
        self.data[3] = new_p

    @eigen.setter
    def eigen(self, new_eigen: ARRAY):
        self.data[2] = new_eigen

    def reset(self):
        self.total_translate = np.zeros(3)
        self.total_rotate = np.eye(3)

    def __getitem__(self, item):
        return self.data[item]

    def __init__(self, gaussian):
        self.recover_data = [item.copy() if type(item) is ARRAY else item for item in gaussian]
        self.data = list(gaussian)
        self.up_axis, self.reflect_up = self.to_positive(self.p)
        self.total_translate = np.zeros(3)
        self.total_rotate = np.eye(3)


class GaussianStatus(GaussianData):

    # copy_constructor
    def copy(self: GaussianStatus, render: vtk.vtkRenderer, view_style: ui_utils.ViewStyle,
             gaussian_id: Optional[Tuple[int, int]] = None, is_selected: Optional[bool] = None) -> GaussianStatus:
        if self.disabled:
            return self
        gaussian_id = self.gaussian_id if gaussian_id is None else gaussian_id
        return GaussianStatus(self.copy_data(), gaussian_id, is_selected or self.is_selected, view_style, render, 1)

    @staticmethod
    def get_new_gaussian() -> vtk.vtkSphereSource:
        return ui_utils.load_vtk_obj(f"{constants.DATA_ROOT}/ui_resources/simple_brick.obj")

    def update_gaussian_transform(self, source):
        phi, mu, eigen, p = self.get_view_data()

        # def replace_mesh(self, mesh: T_Mesh):
        #     vs, faces = mesh
        #     vs, faces = vs.detach().cpu(), faces.detach().cpu()
        #     # vs, faces = mesh_utils.scale_from_ref(mesh, *self.scale)
        #     source = vtk.vtkPolyData()
        #     new_vs_vtk = numpy_support.numpy_to_vtk(vs.numpy())
        #     cells_npy = np.column_stack(
        #         [np.full(faces.shape[0], 3, dtype=np.int64), faces.numpy().astype(np.int64)]).ravel()
        #     vs_vtk, faces_vtk = vtk.vtkPoints(), vtk.vtkCellArray()
        #     vs_vtk.SetData(new_vs_vtk)
        #     faces_vtk.SetCells(faces.shape[0], numpy_support.numpy_to_vtkIdTypeArray(cells_npy))
        #
        #     source.SetPolys(faces_vtk)
        #     self.mapper.SetInputData(source)
        #     self.is_changed = True
        #     if not self.to_init:
        #         self.to_init = True
        #         self.render.AddActor(self.actor)
        transform = vtk.vtkTransform()
        mat = vtk.vtkMatrix4x4()
        p = p * .005
        # p = p * eigen[:, None]
        for i in range(4):
            for j in range(4):
                if i > 2:
                    mat.SetElement(i, j, 0)
                elif j > 2:
                    mat.SetElement(i, j, float(mu[i]))
                    # mat.SetElement(i, j, 0)
                else:
                    mat.SetElement(i, j, p[j, i])
                # mat_t[i, j] = mat.GetElement(i,j)
        mat.SetElement(3, 3, 1)
        transform.SetMatrix(mat)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputData(source)
        transformFilter.SetTransform(transform)
        transformFilter.Update()
        return transformFilter

    def update_gaussian(self):
        if self.disabled:
            return
        # source = self.mapper.GetInput()
        # source.SetPoints(self.init_points)
        # source = self.update_gaussian_transform(source)
        # self.mapper.SetInputConnection(source.GetOutputPort())

    def end_transition(self, transition: ui_utils.Transition) -> bool:
        if self.init_points is None:
            return False
        if transition.transition_type is ui_utils.EditType.Translating:
            self.translate(transition)
            return True
        elif transition.transition_type is ui_utils.EditType.Rotating:
            self.rotate(transition)
            return True
        elif transition.transition_type is ui_utils.EditType.Scaling:
            self.rotate(transition)
            return True
        return False

    def temporary_transition(self, transition: ui_utils.Transition) -> bool:
        if self.init_points is None:
            return False
        source = self.mapper.GetInput()
        vs = self.init_points
        if transition.transition_type is ui_utils.EditType.Translating:
            vs = vs + transition.translation[None, :]
        elif transition.transition_type is ui_utils.EditType.Rotating:
            vs = vs - transition.transition_origin[None, :]
            vs = np.einsum('ad,nd->na', transition.rotation, vs)
            vs = vs + transition.transition_origin[None, :]
        source.GetPoints().SetData(numpy_support.numpy_to_vtk(vs))
        return True

    def get_address(self):
        if self.disabled:
            return f"disabled"
        return self.actor.GetAddressAsString('')

    @staticmethod
    def add_gaussian(render, actor: Optional[vtk.vtkActor]) -> vtk.vtkActor:
        if actor is None:
            actor = vtk.vtkActor()
            mapper = vtk.vtkPolyDataMapper()
            actor.GetProperty().SetOpacity(0.3)
            actor.SetMapper(mapper)
            # source = self.get_new_gaussian()
            # init_points = source.GetPoints()
            # source = self.update_gaussian_transform(source)
            # actor, _ = ui_utils.wrap_mesh(source.GetOutput(), color)
            render.AddActor(actor)
            ui_utils.set_default_properties(actor, (1., 1., .1))
        return actor

    def replace_part(self, part_mesh: Optional[vtk.vtkPolyData]):
        if part_mesh is not None and not self.disabled:
            self.init_points = numpy_support.vtk_to_numpy(part_mesh.GetPoints().GetData())
            points = vtk.vtkPoints()
            points.SetData(numpy_support.numpy_to_vtk(self.init_points))
            part_mesh.SetPoints(points)
            self.mapper.SetInputData(part_mesh)

    def set_color(self):
        if self.disabled:
            return
        properties = self.actor.GetProperty()
        properties.SetOpacity(self.opacity)
        properties.SetColor(*self.color)

    def turn_off(self):
        if self.disabled:
            return
        self.actor.GetProperty().SetOpacity(0)
        self.actor.PickableOff()

    def turn_on(self):
        if self.disabled:
            return
        self.actor.GetProperty().SetOpacity(self.opacity)
        self.actor.PickableOn()

    @property
    def is_not_selected(self):
        return not self.is_selected

    @property
    def disabled(self):
        return self.mapper is None #or self.mapper.GetInput() is None

    def make_symmetric(self, force_include: bool):
        if self.disabled or self.twin is None or (not force_include and self.included != self.twin.included):
            return
        super(GaussianStatus, self).make_symmetric(self.twin)
        if force_include:
            self.included = self.twin.included
        self.update_gaussian()

    def apply_affine(self, button: ui_utils.Buttons, key: str):
        axis = {"left": 0, "right": 0, "up": 2, "down": 2, "a": 1, "z": 1}[key]
        sign = {"left": 1, "right": -1, "up": 1, "down": -1, "a": 1, "z": -1}[key]
        if self.disabled or button not in (ui_utils.Buttons.translate, ui_utils.Buttons.stretch, ui_utils.Buttons.rotate):
            return
        elif button is ui_utils.Buttons.translate:
            vec = np.zeros(3)
            vec[axis] = .01 * sign
            self.translate(vec)
        elif button is ui_utils.Buttons.rotate:
            self.rotate(sign * .1, axis)
        else:
            self.stretch(.01 * sign)
        self.update_gaussian()

    @property
    def opacity(self) -> float:
        if self.is_selected:
            if self.included:
                opacity = self.view_style.opacity + 0.4
            else:
                opacity = self.view_style.opacity
        else:
            if self.included:
                opacity = self.view_style.opacity + 0.2
            else:
                opacity = self.view_style.opacity
        return max(0., min(1., opacity))

    @property
    def color(self) -> Tuple[float, float, float]:
        if self.is_selected:
            return self.view_style.selected_color
        if self.included:
            return self.view_style.included_color
        else:
            return self.view_style.base_color

    def toggle_inclusion(self, included: Optional[bool] = None):
        if self.disabled:
            return
        if included is None:
            included = not self.included
        if included != self.included:
            self.included = not self.included
            self.set_color()

    def toggle_selection(self):
        if self.disabled:
            return
        self.is_selected = not self.is_selected
        self.set_color()

    def reset(self):
        if self.disabled:
            return
        super(GaussianStatus, self).reset()
        self.included = False
        self.is_selected = False
        self.set_color()
        # self.update_gaussian()

    def delete(self, render):
        if not self.disabled:
            render.RemoveActor(self.actor)
        if self.twin is not None:
            self.twin.twin = None

    @property
    def parent_id(self):
        return self.gaussian_id[0]

    @property
    def child_id(self):
        return self.gaussian_id[1]

    @property
    def mapper(self):
        if self.actor is None:
            return None
        return self.actor.GetMapper()

    def __init__(self, gaussian, gaussian_id: Tuple[int, int], is_selected: bool, view_style: ui_utils.ViewStyle,
                 render: vtk.vtkRenderer, normalized_phi: float, actor: Optional[vtk.vtkActor] = None):
        self.view_style = view_style
        super(GaussianStatus, self).__init__(gaussian)
        self.gaussian_id = gaussian_id
        self.twin: Optional[GaussianStatus] = None
        self.init_points = None
        if normalized_phi > 0.001 or actor is not None:
            self.actor = self.add_gaussian(render, actor)
            self.is_selected = is_selected
            self.included = True
            self.set_color()
        else:
            self.actor = None
            self.is_selected = False
            self.included = False
