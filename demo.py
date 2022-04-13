from __future__ import annotations
import abc
from custom_types import *
import vtk
from utils import files_utils, myparse
from ui import ui_utils, ui_controllers, inference_processing, gaussian_status
import options
import vtk.util.numpy_support as numpy_support
import constants
import sys


def to_local(func):
    def inner(self: TransitionController, mouse_pos: Optional[Tuple[int, int]], *args, **kwargs):
        if mouse_pos is not None:
            size_full = self.render.GetRenderWindow().GetScreenSize()
            left, bottom, right, top = viewport = self.render.GetViewport()
            size = size_full[0] * (right - left), size_full[1] * (top - bottom)
            mouse_pos = float(mouse_pos[0]) / size[0] - .5, float(mouse_pos[1]) / size[1] - .5
            mouse_pos = torch.tensor([mouse_pos[0], mouse_pos[1]])
        return func(self, mouse_pos, *args, **kwargs)
    return inner


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
        delta_3d[axis] = delta * self.camera.GetDistance()
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
                - projections[0][(self.moving_axis + 1) % 3] * projections[1][(self.moving_axis + 2) % 3]).sign()
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
        self.origin_mouse, self.dir_2d = mouse_pos, nnf.normalize(dir_2d[:2, :3], p=2, dim=1)
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


class VtkTimerCallback:

    def __init__(self, steps, iren, camera, callback):
        self.scrolling = False
        self.timer_count = 0
        self.steps = steps
        self.iren = iren
        self.camera = camera
        self.pos_x = camera.GetPosition()[0]
        self.callback = callback
        self.timer_id = None

    @staticmethod
    def ease_in_out(alpha: float):
        if alpha < .5:
            return 2 * alpha ** 2
        else:
            return 1 - ((-2 * alpha + 2) ** 4) / 2

    @staticmethod
    def ease_in(alpha: float):
        return alpha ** 2

    @staticmethod
    def ease_out(alpha: float):
        if alpha == 1:
            return 1
        return 1 - 2 ** (-4 * alpha) / (1 - 2 ** (-4))

    def init_scrolling(self):
        if self.timer_id is not None:
            self.iren.DestroyTimer(self.timer_id)
            self.timer_id = None
        self.scrolling = True
        self.timer_count = 0
        self.timer_id = None

    def execute(self, obj, event):
        alpha = self.ease_out(float(self.timer_count + 1) / self.steps)
        # alpha = float(self.timer_count + 1) / self.steps
        pos_x = self.pos_x + 4 * alpha
        self.camera.SetPosition(pos_x, 0, 1)
        self.camera.SetFocalPoint(pos_x, 0, 0)
        iren = obj
        iren.GetRenderWindow().Render()
        self.timer_count += 1
        if self.timer_count == self.steps - 1:
            self.pos_x = self.pos_x + 4
            self.scrolling = False
            self.callback()


class StagedCanvas(ui_utils.CanvasRender, abc.ABC):

    def reset(self):
        self.stage.reset()

    def vote(self, *actors: Optional[vtk.vtkActor]):
        self.stage.vote(*actors)

    @abc.abstractmethod
    def after_draw(self, changed: List[int], select: bool) -> bool:
        raise NotImplementedError

    def aggregate_votes(self, select: bool) -> bool:
        changed = self.stage.aggregate_votes()
        return self.after_draw(changed, select)

    def save(self, root: str):
        self.stage.save(root, ui_controllers.filter_by_selection)

    def __init__(self, opt, viewport:Tuple[float, float, float, float], render_window: vtk.vtkRenderWindow,
                 bg_color: ui_utils.RGB_COLOR, stroke_color: Optional[ui_utils.RGBA_COLOR]):
        super(StagedCanvas, self).__init__(viewport, render_window, bg_color, stroke_color)
        self.stage = ui_controllers.GmmMeshStage(opt, ['-1', '-1', '-1'], self, -1, ui_utils.ViewStyle((255, 255, 255),
                                                                                      (255, 255, 255), ui_utils.bg_target_color, 1))


class RenderPop(StagedCanvas):

    def after_draw(self, changed: List[int], select: bool) -> bool:
        changed = list(filter(lambda x: not self.stage.gmm[x].disabled and self.stage.gmm[x].included != select, changed))
        for item in changed:
            self.stage.gmm[item].toggle_inclusion()
            self.source_stage.gmm[item].toggle_inclusion()
        self.afetr_draw_callback(self.source_stage, changed)
        return len(changed) > 0

    def replace_stage(self, mesh_gmm: ui_controllers.GmmMeshStage):
        self.change_brush(mesh_gmm.view_style.stroke_color)
        self.stage.remove_all()
        self.source_stage = mesh_gmm
        self.stage.view_style = mesh_gmm.view_style
        self.stage.add_gaussians(mesh_gmm.gmm)
        for gaussian_ref, gaussian in zip(mesh_gmm.gmm, self.stage.gmm):
            vs = None
            vs_vtk = vtk.vtkPoints()
            if not gaussian.disabled and gaussian_ref.actor.GetMapper().GetInput() is not None:
                gaussian.toggle_inclusion(gaussian_ref.included)
                ref_part = gaussian_ref.actor.GetMapper().GetInput()
                part_mesh = vtk.vtkPolyData()
                if vs is None and ref_part is not None:
                    vs = numpy_support.vtk_to_numpy(ref_part.GetPoints().GetData()).copy()
                    vs[:, 0] -= mesh_gmm.offset * 2
                    vs_vtk.SetData(numpy_support.numpy_to_vtk(vs))
                part_mesh.SetPoints(vs_vtk)
                part_mesh.SetPolys(ref_part.GetPolys())
                gaussian.mapper.SetInputData(part_mesh)
        self.maximize()

    def change_viewport(self, viewport: Tuple[float, float, float, float]):
        self.viewport = viewport
        self.canvas_render.SetViewport(*viewport)
        self.SetViewport(*viewport)
        self.resize_event_(self.render_window)

    def toggle_win(self, button_, event):
        if button_.GetRepresentation().GetState() == 0:
            new_viewport = self.min_viewport
        else:
            new_viewport = self.max_viewport
        self.change_viewport(new_viewport)

    def maximize(self):
        if self.button_minimize.GetRepresentation().GetState() == 0:
            self.button_minimize.GetRepresentation().SetState(1)
            self.change_viewport(self.max_viewport)

    def __init__(self, opt, min_viewport:Tuple[float, float, float, float], max_viewport: Tuple[float, float, float, float],
                 render_window: vtk.vtkRenderWindow, iren, bg_color: ui_utils.RGB_COLOR, stroke_color: ui_utils.RGBA_COLOR,
                 afetr_draw_callback: Callable[[ui_controllers.GmmMeshStage, List[int]], None]):
        super(RenderPop, self).__init__(opt, min_viewport, render_window, bg_color, stroke_color)
        self.afetr_draw_callback = afetr_draw_callback
        self.min_viewport = min_viewport
        self.max_viewport = max_viewport
        self.render_window = render_window
        self.button_minimize = ui_utils.ImageButton(
            [f'{constants.UI_RESOURCES}icons-14.png', f'{constants.UI_RESOURCES}icons-13.png'], iren, self, (0.015, 0.015),
            (0.001, 0.8), on_click=self.toggle_win)
        self.source_stage: Optional[ui_controllers.GmmMeshStage] = None
        self.GetActiveCamera().SetPosition(0, 0, 4)


class RenderMain(StagedCanvas):

    def toggle_edit_direction(self, direction: ui_utils.EditDirection):
        self.transition_controller.toggle_edit_direction(direction)

    def clear_selection(self) -> bool:
        is_changed = False
        for gaussian in self.selected_gaussians:
            gaussian.toggle_selection()
            is_changed = True
        return is_changed

    @property
    def selected_gaussians(self) -> Iterable[gaussian_status.GaussianStatus]:
        return filter(lambda x: x.is_selected, self.stage.gmm)

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
        center = torch.from_numpy(np.stack(center, axis=0).mean(0))
        # center = torch.zeros(3)
        self.transition_controller.init_transition(mouse_pos, center, transition_type)

    def reset(self):
        self.stage.remove_all()
        self.stage_mapper = {}

    def make_twins(self, toggled: List[gaussian_status.GaussianStatus], new_addresses: List[str]):
        if len(new_addresses) == 2:
            self.stage.make_twins(*new_addresses)
        else:
            if toggled[0].twin is not None and toggled[0].twin.get_address() in self.stage_mapper:
                self.stage.make_twins(new_addresses[0], self.stage_mapper[toggled[0].twin.get_address()])

    def update_gmm(self, stage: ui_controllers.GmmMeshStage, changed: List[int]):
        for item in changed:
            is_toggled, toggled = stage.toggle_inclusion_by_id(item, self.is_draw)
            if is_toggled:
                if toggled[0].included:
                    new_addresses = self.stage.add_gaussians(toggled)
                    for gaussian, new_address in zip(toggled, new_addresses):
                        self.stage_mapper[gaussian.get_address()] = new_address
                    self.make_twins(toggled, new_addresses)
                else:
                    addresses = [gaussian.get_address() for gaussian in toggled]
                    addresses = list(filter(lambda x: x in self.stage_mapper, addresses))
                    self.stage.remove_gaussians([self.stage_mapper[address] for address in addresses])
                    for address in addresses:
                        del self.stage_mapper[address]

    def update_mesh(self, res=128):
        if self.model_process is not None:
            self.model_process.get_mesh(res)
            return True
        return False

    def after_draw(self, changed: List[int], select: bool) -> bool:
        changed = list(filter(lambda x: not self.stage.gmm[x].disabled and self.stage.gmm[x].is_selected != select, changed))
        for item in changed:
            self.stage.gmm[item].toggle_selection()
        return False

    def replace_mesh(self):
        if self.model_process is not None:
            self.model_process.replace_mesh()

    def exit(self):
        if self.model_process is not None:
            self.model_process.exit()

    def __init__(self, opt, viewport:Tuple[float, float, float, float], samples_root, render_window: vtk.vtkRenderWindow,
                 bg_color: ui_utils.RGB_COLOR, stroke_color: Optional[ui_utils.RGBA_COLOR], with_model: bool):
        super(RenderMain, self).__init__(opt, viewport, render_window, bg_color, stroke_color)
        if with_model:
            self.model_process = inference_processing.InferenceProcess(opt, self.stage.replace_mesh,
                                                                       self.stage.get_gmm,
                                                                       samples_root)
        else:
            self.model_process = None
        self.stage_mapper: Dict[str, str] = {}
        self.transition_controller = TransitionController(self)
        self.GetActiveCamera().SetPosition(3, 1, -3)


class MeshScroller(vtk.vtkRenderer):

    def reset(self):
        self.meshes.reset()

    def set_int_viewport(self, win_size) -> Tuple[int, int, int, int]:
        w, h = win_size
        return int(self.viewport[0] * w), int(self.viewport[1] * h), int(self.viewport[2] * w), int(self.viewport[3] * h)

    def set_camera(self):
        camera = self.GetActiveCamera()
        camera.ParallelProjectionOn()
        camera.SetPosition(len(self.meshes), 0, 1)
        camera.SetFocalPoint(len(self.meshes), 0, 0)

    def resize_event(self, obj, event):
        self.viewport_ren = self.set_int_viewport(obj.GetSize())

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

    def move_mesh(self):
        self.meshes.move_mesh_to_end(self.actor_ptr)

    def move_camera(self):
        camera = self.GetActiveCamera()
        pos_x = camera.GetPosition()[0]
        camera.SetPosition(pos_x + 2, 0, 1)
        camera.SetFocalPoint(pos_x + 2, 0, 0)

    def end_scroll_callback(self):
        if len(self.cb_stack) > 0:
            self.scroll(*self.cb_stack.pop(0))
        else:
            self.iren.DestroyTimer(self.cb.timer_id)

    def scroll(self, button, event):
        if self.cb.scrolling:
            self.cb_stack.append((button, event))
            return
        self.move_mesh()
        self.actor_ptr = (self.actor_ptr + 1) % (len(self.meshes))
        self.move_mesh()
        # self.move_camera()
        self.cb.init_scrolling()
        self.actor_ptr = (self.actor_ptr + 1) % (len(self.meshes))
        self.cb.timer_id = self.iren.CreateRepeatingTimer(30)

    def pick(self, actor_address: str) -> Optional[ui_controllers.GmmMeshStage]:
        return self.meshes.pick(actor_address)

    def save(self, root: str, gmms):
        self.meshes.save(root, gmms)

    def __init__(self, opt, viewport: Tuple[float, float, float, float], render_window: vtk.vtkRenderWindow, iren,
                 bg_color: ui_utils.RGB_COLOR, samples_root: str):
        super(MeshScroller, self).__init__()
        self.iren = iren
        self.SetViewport(*viewport)
        self.viewport = viewport
        self.viewport_ren = self.set_int_viewport(render_window.GetSize())
        render_window.AddObserver(vtk.vtkCommand.WindowResizeEvent, self.resize_event)
        self.SetBackground(*ui_utils.rgb_to_float(bg_color))
        self.actor_ptr = 0
        shape_paths = files_utils.collect(samples_root, '.obj')
        view_styles = ui_utils.get_view_styles(len(shape_paths), False)
        self.meshes = ui_controllers.MeshGmmStatuses(opt, shape_paths, self, view_styles, False)
        render_window.AddRenderer(self)
        self.set_camera()
        self.cb = VtkTimerCallback(30, self.iren, self.GetActiveCamera(), self.end_scroll_callback)
        self.iren.AddObserver('TimerEvent', self.cb.execute)
        self.cb_stack = []
        self.button_slide = ui_utils.ImageButton([f'{constants.UI_RESOURCES}icons-06.png'], iren, self, (0.05, 0.02),
                                                 (0.001, 0.14), on_click=self.scroll)


class InteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

    class MouseStatus:

        def update(self, pos: Tuple[int, int], selected_view: int) -> bool:
            is_changed = selected_view != self.selected_view or (pos[0] - self.last_pos[0]) ** 2 > 4 and (
                        pos[1] - self.last_pos[1]) ** 2 > 4
            self.selected_view = selected_view
            self.last_pos = pos
            return is_changed

        def __init__(self):
            self.last_pos = (0, 0)
            self.selected_view = 0

    def on_key_press(self, obj, event):
        key = self.interactor.GetKeySym()
        if type(key) is not str:
            return
        key: str = key.lower()
        if self.pondering:
            if key in ('g', 'r'):
                self.edit_status = {'g': ui_utils.EditType.Translating,
                                    'r': ui_utils.EditType.Rotating,
                                    's': ui_utils.EditType.Scaling}[key]
                click_pos = self.interactor.GetEventPosition()
                self.render_main.init_transition(click_pos, self.edit_status)
            elif key == 'escape':
                if self.render_main.clear_selection():
                    self.interactor.Render()
        elif self.in_transition:
            if key == 'escape':
                self.render_main.temporary_transition()
                self.interactor.Render()
                self.edit_status = ui_utils.EditType.Pondering
            elif key in ('x', 'y', 'z'):
                self.render_main.toggle_edit_direction({'x': ui_utils.EditDirection.X_Axis,
                                                   'y': ui_utils.EditDirection.Y_Axis,
                                                   'z': ui_utils.EditDirection.Z_Axis}[key])
        if key == 'return' or key == 'kp_enter':
            gmms = self.render_main.stage.get_gmm()
            self.mesh_scroller.save(self.root, gmms)
            # self.render_main.save(self.root)

        if key == 'control_l':
            self.render_main.replace_mesh()
            self.interactor.Render()

        return

    def mouse_out_scroller(self):
        click_pos = self.interactor.GetEventPosition()
        val = click_pos[1] / self.interactor.GetRenderWindow().GetSize()[1]
        return click_pos, val > self.mesh_scroller.GetViewport()[-1] + 0.01

    @property
    def interactor(self):
        return self.GetInteractor()

    def replace_pop(self, actor):
        if actor is not None:
            address = actor.GetAddressAsString('')
            mesh_gmm = self.mesh_scroller.pick(address)
            self.render_pop.replace_stage(mesh_gmm)
            self.interactor.Render()

    def replace_render(self):
        self.OnLeftButtonDown()
        self.OnLeftButtonUp()
        return self.GetCurrentRenderer()

    def left_button_press_event(self, obj, event):
        cur_render = self.replace_render()
        if cur_render is self.mesh_scroller:
            click_pos = self.interactor.GetEventPosition()
            picker = vtk.vtkPropPicker()
            picker.Pick(click_pos[0], click_pos[1], 0, cur_render)
            self.replace_pop(picker.GetActor())
        elif self.in_transition:
            click_pos = self.interactor.GetEventPosition()
            if self.render_main.end_transition(click_pos):
                self.render_main.update_mesh()
            self.edit_status = ui_utils.EditType.Pondering
        elif self.mouse_out_scroller()[1]:
            super(InteractorStyle, self).OnLeftButtonDown()

    def left_button_release_event(self, obj, event):
        if self.mouse_out_scroller()[1]:
            super(InteractorStyle, self).OnLeftButtonUp()

    def right_button_release_event(self, obj, event):
        if self.marking:
            self.draw_end()
            self.edit_status = ui_utils.EditType.Pondering

    # def get_cur_view(self):
    #     cur_render = self.GetCurrentRenderer()
    #     if cur_render is None:
    #         return cur_render
    #     return cur_render

    def update_default(self):
        click_pos = self.interactor.GetEventPosition()
        picker = vtk.vtkPropPicker()
        cur_render = self.GetCurrentRenderer()
        if cur_render != self.selected_view:
            # self.SetDefaultRenderer(self.renders[cur_view])
            self.selected_view = cur_render
        elif cur_render is None:
            return None, False, click_pos
        picker.Pick(click_pos[0], click_pos[1], 0, cur_render)
        return picker, self.mouse_status.update((click_pos[0], click_pos[1]), cur_render), click_pos

    def draw_end(self):
        self.draw_view.clear()
        if self.draw_view.aggregate_votes(self.select_mode):
            self.render_main.update_mesh()
        self.interactor.Render()

    def right_button_press_event(self, obj, event):
        self.OnRightButtonDown()
        self.OnRightButtonUp()
        _ = self.update_default()
        if self.pondering and self.mouse_out_scroller()[1]:
            self.draw_view = self.selected_view
            self.edit_status = ui_utils.EditType.Marking
            # self.draw_init()
            # super(InteractorStyle, self).OnRightButtonDown()
        return

    def on_mouse_wheel_backward(self, obj, event):
        if self.mouse_out_scroller()[1]:
            self.OnMouseWheelBackward()
        return

    def on_mouse_wheel_forward(self, obj, event):
        if self.mouse_out_scroller()[1]:
            super(InteractorStyle, self).OnMouseWheelForward()
        return

    def middle_button_press_event(self, obj, event):
        if self.replace_render() != self.mesh_scroller:
            super(InteractorStyle, self).OnMiddleButtonDown()
        return

    def middle_button_release_event(self, obj, event):
        if self.replace_render() != self.mesh_scroller:
            super(InteractorStyle, self).OnMiddleButtonUp()
        return

    def get_trace(self):
        picker, is_changed, pos_2d = self.update_default()
        if picker is not None:
            points = self.draw_view.draw(pos_2d)
            actors = []
            for point in points:
                picker.Pick(point[0], point[1], 0, self.draw_view)
                actors.append(picker.GetActor())
            self.draw_view.vote(*actors)
            self.interactor.Render()
        return 0

    def on_mouse_move(self, obj, event):
        if self.marking:
            self.get_trace()
        elif self.in_transition:
            click_pos = self.interactor.GetEventPosition()
            if self.render_main.temporary_transition(click_pos):
                self.interactor.Render()
        elif self.GetCurrentRenderer() != self.mesh_scroller:
            super(InteractorStyle, self).OnMouseMove()

    def add_buttons(self, interactor):

        def toggle_select_mode(button, __):
            self.select_mode = button.GetRepresentation().GetState() == 0
            self.render_pop.set_brush(self.select_mode)
            self.render_main.set_brush(self.select_mode)

        def reset(_, __):
            self.render_pop.reset()
            self.mesh_scroller.reset()
            self.render_main.reset()
            self.interactor.Render()

        button_pencil = ui_utils.ImageButton([f"{constants.UI_RESOURCES}icons-03.png", f"{constants.UI_RESOURCES}icons-04.png"],
                                             interactor, self.render_main, (.08, .08), (0.01, 0.1), toggle_select_mode, full_size=(1., .85))
        button_reset = ui_utils.ImageButton([f"{constants.UI_RESOURCES}icons-05.png"], interactor, self.render_main,
                                            (.08, .08), (0.01, 0.98), reset, full_size=(1., .85))
        return button_pencil, button_reset

    @property
    def marking(self) -> bool:
        return self.edit_status == ui_utils.EditType.Marking

    @property
    def pondering(self) -> bool:
        return self.edit_status == ui_utils.EditType.Pondering

    @property
    def translating(self) -> bool:
        return self.edit_status == ui_utils.EditType.Translating

    @property
    def rotating(self) -> bool:
        return self.edit_status == ui_utils.EditType.Rotating

    @property
    def scaling(self) -> bool:
        return self.edit_status == ui_utils.EditType.Scaling

    @property
    def in_transition(self):
        return self.translating or self.rotating or self.scaling

    def init_observers(self):
        self.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self.left_button_press_event)
        self.AddObserver(vtk.vtkCommand.LeftButtonReleaseEvent, self.left_button_release_event)
        self.AddObserver(vtk.vtkCommand.RightButtonReleaseEvent, self.right_button_release_event)
        self.AddObserver(vtk.vtkCommand.RightButtonPressEvent, self.right_button_press_event)
        self.AddObserver(vtk.vtkCommand.MouseWheelBackwardEvent, self.on_mouse_wheel_backward)
        self.AddObserver(vtk.vtkCommand.MouseWheelForwardEvent, self.on_mouse_wheel_forward)
        self.AddObserver(vtk.vtkCommand.MiddleButtonPressEvent, self.middle_button_press_event)
        self.AddObserver(vtk.vtkCommand.MiddleButtonReleaseEvent, self.middle_button_release_event)
        self.AddObserver(vtk.vtkCommand.MouseMoveEvent, self.on_mouse_move)
        self.AddObserver(vtk.vtkCommand.KeyPressEvent, self.on_key_press)
        self.AddObserver(vtk.vtkCommand.CharEvent, lambda _, __: None)

    def __init__(self, opt: options.Options, mesh_scroller: MeshScroller, render_pop: RenderPop, render_main: RenderMain, interactor):
        super(InteractorStyle, self).__init__()
        self.mouse_status = self.MouseStatus()
        self.mesh_scroller = mesh_scroller
        self.render_main = render_main
        self.edit_status = ui_utils.EditType.Pondering
        self.edit_direction = ui_utils.EditDirection.X_Axis
        self.select_mode = True
        self.selected_view: Optional[StagedCanvas] = None
        self.draw_view: Optional[StagedCanvas] = None
        self.render_pop = render_pop
        self.init_observers()
        self.buttons = self.add_buttons(interactor)
        self.root = f"{constants.UI_OUT}{files_utils.get_time_name(opt.info)}/"


def run(model_tag, samples_dir, with_model: bool = True):
    opt = options.Options(tag=model_tag).load()
    samples_root = f'{opt.cp_folder}/{samples_dir}/occ'
    min_viewport = (0.65, 0.15, 1., 0.17)
    max_viewport = (0.65, .15, 1., .6)
    render_window = vtk.vtkRenderWindow()
    # shape_ids = map(lambda x: x[1], files_utils.collect(samples_root, '.obj'))
    render_window.SetNumberOfLayers(3)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    renderer_main = RenderMain(opt, (0., .15, 1., 1.), samples_root, render_window, ui_utils.bg_source_color,
                                          list(ui_utils.bg_target_color) + [200], with_model)
    mesh_scroller = MeshScroller(opt, (0., 0., 1., .15), render_window, interactor, ui_utils.bg_target_color,
                                 samples_root)
    renderer_pop = RenderPop(opt, min_viewport, max_viewport, render_window, interactor, ui_utils.bg_stage_color,
                             list(ui_utils.bg_source_color) + [200], renderer_main.update_gmm)
    render_window.Render()
    interactor.Initialize()
    style = InteractorStyle(opt, mesh_scroller, renderer_pop, renderer_main, interactor)
    interactor.SetInteractorStyle(style)
    render_window.Render()
    interactor.Start()
    del interactor
    del render_window
    renderer_main.exit()
    return 0


def main():
    for_parser = {'--model_name': {'default': 'chairs_large', 'type': str},
                  '--shape_dir': {'default': 'inversion', 'type': str}}
    args = myparse.parse(for_parser)
    run(args['model_name'], args['shape_dir'])
    return 0


if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    main()
