from __future__ import annotations
import abc
import vtk
from custom_types import *
from ui import ui_utils, ui_controllers
import colorsys
import options


class CustomTextWidget(vtk.vtkTextWidget):

    def set_to_start(self):
        # self.EnabledOff()
        h, s, v = colorsys.rgb_to_hsv(*ui_utils.button_color)
        self.GetTextActor().GetTextProperty().SetColor(*ui_utils.rgb_to_float(colorsys.hsv_to_rgb(h, .3, 255)))
        # self.EnabledOn()
        self.GetInteractor().Render()
        self.on = False

    def set_to_stop(self):
        # self.EnabledOff()
        h, s, v = colorsys.rgb_to_hsv(*ui_utils.button_color)
        self.GetTextActor().GetTextProperty().SetColor(*ui_utils.rgb_to_float(colorsys.hsv_to_rgb(h, .9, 255)))
        # self.EnabledOn()
        self.GetInteractor().Render()
        self.on = True

    def left_button_press_event(self, obj, event):
        self.on_click(self)

    def hover(self, obj, event):
        print(self.b_type.value)

    def toggle_selection(self):
        if self.on:
            self.set_to_start()
        else:
            self.set_to_stop()

    def __init__(self, b_type: ui_utils.Buttons, position, size, interactor: vtk.vtkRenderWindowInteractor, ren: vtk.vtkRenderer, on_click):
        super(CustomTextWidget, self).__init__()
        self.ren = ren
        self.on_click = on_click
        self.SetInteractor(interactor)
        text_actor = vtk.vtkTextActor()
        text_representation = vtk.vtkTextRepresentation()
        self.SetRepresentation(text_representation)
        self.SetTextActor(text_actor)
        self.GetRepresentation().SetRenderer(self.ren)
        self.SetCurrentRenderer(self.ren)
        self.GetTextActor().SetInput(b_type.value)
        self.GetRepresentation().GetPositionCoordinate().SetValue(*position)
        self.set_to_start()
        # self.Set
        self.SelectableOn()
        self.SetResizable(0)
        self.EnabledOn()
        self.On()
        self.AddObserver(vtk.vtkCommand.EndInteractionEvent, self.left_button_press_event)
        self.AddObserver(vtk.vtkCommand.HoverEvent, self.hover)
        self.on = False
        self.b_type = b_type
        self.GetRepresentation().GetPosition2Coordinate().SetValue(*size)

    def get_selection(self) -> ui_utils.Buttons:
        return self.select


class InteractorStyle(vtk.vtkInteractorStyleTrackballCamera, abc.ABC):

    @abc.abstractmethod
    def init_renders(self) -> List[vtk.vtkRenderer]:
        raise NotImplementedError

    @abc.abstractmethod
    def button_release_event(self, to_do):
        raise NotImplementedError

    @abc.abstractmethod
    def draw_end(self):
        raise NotImplementedError

    @property
    def stage(self) -> ui_controllers.GmmMeshStage:
        return self.status.main_stage

    @property
    def canvas(self):
        if self.draw_view == -1:
            return None
        return self.renders[self.draw_view]

    @property
    def interactor(self):
        return self.GetInteractor()

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

    def change_opacity(self, sphere_ids: List[int], is_left: bool, on: bool):
        actors = self.renders[self.selected_view].GetActors()
        for sphere_ids in sphere_ids:
            properties = actors.GetItemAsObject(sphere_ids).GetProperty()
            properties.SetOpacity(float(on))

    def slider_event(self, slider, event):
        value = slider.GetRepresentation().GetValue()
        value = value / 100.
        self.stage.set_opacity(value)
        return

    def get_cur_view(self):
        cur_render = self.GetCurrentRenderer()
        if cur_render is None:
            return -1, cur_render
        return self.rend_to_view[cur_render.GetAddressAsString("")], cur_render

    def update_default(self):
        click_pos = self.interactor.GetEventPosition()
        picker = vtk.vtkPropPicker()
        cur_view, cur_render = self.get_cur_view()
        if cur_view != self.selected_view:
            # self.SetDefaultRenderer(self.renders[cur_view])
            self.selected_view = cur_view
        elif cur_render is None:
            return None, False, click_pos
        picker.Pick(click_pos[0], click_pos[1], 0, cur_render)
        return picker, self.mouse_status.update((click_pos[0], click_pos[1]), cur_view), click_pos

    def get_trace(self):
        picker, is_changed, pos_2d = self.update_default()
        if picker is not None:
            points = self.canvas.draw(pos_2d)
            actors = []
            for point in points:
                picker.Pick(point[0], point[1], 0, self.canvas)
                actors.append(picker.GetActor())
            self.status.vote(*actors)
            self.interactor.Render()
        return 0

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
                self.status.init_transition(click_pos, self.edit_status)
            elif key == 'escape':
                if self.status.clear_selection():
                    self.interactor.Render()
        elif self.in_transition:
            if key == 'escape':
                self.status.temporary_transition()
                self.interactor.Render()
                self.edit_status = ui_utils.EditType.Pondering
            elif key in ('x', 'y', 'z'):
                self.status.toggle_edit_direction({'x': ui_utils.EditDirection.X_Axis,
                                                   'y': ui_utils.EditDirection.Y_Axis,
                                                   'z': ui_utils.EditDirection.Z_Axis}[key])
        if key == 'return' or key == 'kp_enter':
            self.status.save()
        # elif key in ("left", "right", "up", "down", "a", "z"):
        #     if self.status.update_gmm(ui_utils.Buttons.translate, key):
        #         self.status.update_mesh()
        #         self.interactor.Render()
            # axis = {"left": 0, "right", "up", "down", "a", "z")

        elif key == 'control_l':
            self.status.replace_mesh()
            self.interactor.Render()

        return

    def draw_init(self):
        self.status.init_draw(self.draw_view)

    def left_button_release_event(self, obj, event):
        if self.pondering:
            # self.button_release_event(self.status.event_manger)
            self.OnLeftButtonUp()

    def left_button_press_event(self, obj, event):
        if self.pondering:
            _ = self.update_default()
            self.OnLeftButtonDown()
        elif self.in_transition:
            click_pos = self.interactor.GetEventPosition()
            if self.status.end_transition(click_pos):
                self.status.update_mesh()
            self.edit_status = ui_utils.EditType.Pondering

    def right_button_release_event(self, obj, event):
        if self.marking:
            self.draw_end()
            self.edit_status = ui_utils.EditType.Pondering

    def right_button_press_event(self, obj, event):
        self.OnRightButtonDown()
        self.OnRightButtonUp()
        _ = self.update_default()
        if -1 < self.selected_view and self.pondering:
            self.draw_view = self.selected_view
            self.draw_init()
            self.edit_status = ui_utils.EditType.Marking

    def on_mouse_move(self, obj, event):
        if self.marking:
            self.get_trace()
        elif self.in_transition:
            click_pos = self.interactor.GetEventPosition()
            if self.status.temporary_transition(click_pos):
                self.interactor.Render()
        else:
            super(InteractorStyle, self).OnMouseMove()

    def toggle_symmetric(self, _, __):
        self.status.toggle_symmetric()
        self.interactor.Render()

    def toggle_arrows(self, button, __):
        self.status.switch_arrows({0: ui_utils.Buttons.translate, 1: ui_utils.Buttons.rotate}[button.GetRepresentation().GetState()])
        self.interactor.Render()

    def reset(self, obj, event):
        self.status.reset()
        self.interactor.Render()

    def toggle_select_mode(self, button, __):
        self.select_mode = button.GetRepresentation().GetState() == 0
        self.status.set_brush(self.select_mode)
        self.interactor.Render()

    def add_buttons(self, interactor):
        button_pencil = ui_utils.ImageButton(["../assets/ui_resources/icons-03.png", "../assets/ui_resources/icons-04.png"],
                                                  ui_utils.bg_menu_color, interactor, (.1, .1), (0.24, 0.15),
                                                  interactor.GetRenderWindow(), self.toggle_select_mode)
        button_symmetric = ui_utils.ImageButton(["../assets/ui_resources/icons-08.png", "../assets/ui_resources/icons-09.png"],
                                                     ui_utils.bg_menu_color, interactor, (.1, .1), (0.3, 0.15),
                                                     interactor.GetRenderWindow(), self.toggle_symmetric)
        button_reset = ui_utils.ImageButton(["../assets/ui_resources/icons-05.png"], ui_utils.bg_menu_color, interactor,
                                                 (.1, .1), (0.35, .15), interactor.GetRenderWindow(), self.reset)
        slider_widget, _ = ui_utils.make_slider(interactor, self.slider_event)
        return button_pencil, button_symmetric, button_reset, slider_widget

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

    def __init__(self, opt: options.Options, status: ui_controllers.MeshGmmStatuses, interactor: vtk.vtkRenderWindowInteractor):
        super(InteractorStyle, self).__init__()
        self.mouse_status = self.MouseStatus()
        self.opt = opt
        self.edit_status = ui_utils.EditType.Pondering
        self.edit_direction = ui_utils.EditDirection.X_Axis
        self.select_mode = True
        self.AddObserver(vtk.vtkCommand.KeyPressEvent, self.on_key_press)
        self.AddObserver(vtk.vtkCommand.MouseMoveEvent, self.on_mouse_move)
        self.AddObserver(vtk.vtkCommand.LeftButtonReleaseEvent, self.left_button_release_event)
        self.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self.left_button_press_event)
        self.AddObserver(vtk.vtkCommand.RightButtonReleaseEvent, self.right_button_release_event)
        self.AddObserver(vtk.vtkCommand.RightButtonPressEvent, self.right_button_press_event)
        self.AddObserver(vtk.vtkCommand.CharEvent, lambda _, __: None)
        self.status = status
        self.renders = self.init_renders()
        self.rend_to_view = {render.GetAddressAsString(""): i for i, render in enumerate(self.renders)}
        self.selected_view = -1
        self.draw_view = -1
        self.buttons = self.add_buttons(interactor)


class MixInteractorStyle(InteractorStyle):

    def draw_end(self):
        self.canvas.clear()
        if self.status.aggregate_votes(self.select_mode):
            self.status.update_mesh()
        self.interactor.Render()

    def init_renders(self) -> List[vtk.vtkRenderer]:
        return [stage.render for stage in self.status.stages] + [self.status.main_stage.render]

    # def on_key_press(self, obj, event):
    #     key = self.interactor.GetKeySym()
    #     if type(key) is not str:
    #         return
    #     key: str = key.lower()
    #     if key == 'return' or key == 'kp_enter':
    #         self.status.save()
    #     if key == 'space':
    #         self.mark_ready = True
    #     #     if self.status.update_mesh():
    #     #         self.interactor.Render()
    #     # # if key == 'f' or key == 'F':
    #     #     self.status.flip(1 - int(self.is_left_viewport))
    #     # if key == 'f':
    #     #     self.status.flip(1 - int(self.is_left_viewport))
    #     elif key in ("left", "right", "up", "down", "a", "z"):
    #         if self.status.update_gmm(ui_utils.Buttons.translate, key):
    #             self.status.update_mesh()
    #             self.interactor.Render()
    #         # axis = {"left": 0, "right", "up", "down", "a", "z")
    #     elif key == 'control_l':
    #         self.status.replace_mesh()
    #         self.interactor.Render()
    #     # if key == 'b' or key == 'B':
    #     #     self.mesh_stage_left.replace_mesh(files_utils.load_mesh(
    #     #         r"C:\Users\t-amhert\PycharmProjects\sdf_gmm\assets\checkpoints\sdformer_sn_chairs_deep\sdfs/samples_953"))

    def button_release_event(self, to_do):
        picker, is_changed, _ = self.update_default()
        if picker is not None and not is_changed:
            picked_actor = picker.GetActor()
            if picked_actor:
                object_id = picked_actor.GetAddressAsString('')
                if to_do(object_id):
                    self.status.update_mesh(128)


class SingleInteractorStyle(InteractorStyle):

    def draw_end(self):
        self.canvas.clear()
        if self.status.aggregate_votes(self.select_mode):
            self.interactor.Render()

    def init_renders(self) -> List[vtk.vtkRenderer]:
        return [stage.render for stage in self.status.stages]

    def button_release_event(self, to_do):
        picker, is_changed, _ = self.update_default()
        if picker is not None and not is_changed:
            picked_actor = picker.GetActor()
            if picked_actor:
                mapper = picked_actor.GetMapper()
                object_id = mapper.GetAddressAsString('')
                if to_do(object_id):
                    self.status.update_mesh(128)
